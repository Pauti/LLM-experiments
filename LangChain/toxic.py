
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

_toxicity_model_path = "martin-ha/toxic-comment-model"

model_path = _toxicity_model_path
_toxicity_tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
_toxicity_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_toxicity_tokenizer
    )

text="You really suck"
result = _toxicity_pipeline(
        text, truncation=True, max_length=_toxicity_tokenizer.model_max_length
    )
print(result)


text="I love you"
result = _toxicity_pipeline(
        text, truncation=True, max_length=_toxicity_tokenizer.model_max_length
    )
print(result)