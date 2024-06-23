from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import evaluate
import os
import torch

cache_dir = "/mnt/storage/tmp"
os.environ["HF_CACHE_DIR"] = cache_dir

common_voice = DatasetDict()
common_voice = load_dataset("VladS159/common_voice_17_0_romanian_speech_synthesis", token="xxxxxxxxxxxxxx", cache_dir = "/mnt/storage/tmp")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium", cache_dir = "/mnt/storage/tmp")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="ro", task="transcribe", cache_dir = "/mnt/storage/tmp")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="ro", task="transcribe", cache_dir = "/mnt/storage/tmp")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
    
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
        
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium", use_cache=False, cache_dir = "/mnt/storage/tmp")

model.config.forced_decoder_ids = None
model.generate = partial(model.generate, language="ro", task="transcribe")
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="/mnt/storage/tmp/whisper_medium-ro_3_gpus_10000_steps",
    per_device_train_batch_size=10,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=1000,
    max_steps=10000,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":True},
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=10,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=500,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="Whisper_medium_ro_VladS_10000_steps_3_gpus",
    hub_token="xxxxxxxxxxxxxx",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

kwargs = {
    "dataset_tags": "VladS159/common_voice_17_0_romanian_speech_synthesis",
    "dataset": "Common Voice 17.0 + Romanian speech synthesis",
    "dataset_args": "config: ro, split: test",
    "language": "ro",
    "model_name": "Whisper Medium Ro - Sarbu Vlad -> 3 gpus",
    "finetuned_from": "openai/whisper-medium",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

trainer.push_to_hub(**kwargs)
#reference https://huggingface.co/blog/fine-tune-whisper
