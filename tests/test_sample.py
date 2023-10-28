from transformers import AutoModelForCausalLM, AutoTokenizer
from toolformer.data_generator import DataGenerator
from toolformer.api import CalculatorAPI
from toolformer.prompt import calculator_prompt
from toolformer.utils import yaml2dict

config = yaml2dict('configs/default.yaml')
calculator_api = CalculatorAPI(
    "Calculator", calculator_prompt,
    sampling_threshold=0.2, filtering_threshold=0.2
)

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

text = "Out of 1400 participants, 400 (or 29%) passed the test."
apis = [calculator_api]
generator = DataGenerator(config, model, tokenizer, apis=apis)
augumented_text_ids = generator.generate(text)
print(augumented_text_ids)
print(tokenizer.decode(augumented_text_ids[0][0], skip_special_tokens=True))
