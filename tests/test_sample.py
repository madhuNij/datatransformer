from transformers import AutoModelForCausalLM, AutoTokenizer
from toolformer.data_generator import DataGenerator
from toolformer.api import CalendarAPI, WolframeAPI, CalculatorAPI
from toolformer.prompt import calendar_prompt, wolframe_prompt, calculator_prompt
from toolformer.utils import yaml2dict

config = yaml2dict('configs/default.yaml')
calendar_api = CalendarAPI(
    "Calendar", calendar_prompt,
    sampling_threshold=0.2, filtering_threshold=0.2
)

WOLFRAME_API_KEY = 'XT7XK3-P9EQETTLJ2'
wolframe_api = WolframeAPI("Wolframe", wolframe_prompt, api_key=WOLFRAME_API_KEY)
calculator_api = CalculatorAPI("Calculator", calculator_prompt)

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

text = "The complex conjugate of 4 + 7i is 4 - 7i."
text3 = "Given a sequence of numbers: 21.3, 38.4, 12.7, 41.6. The mean is 28.5"
text4 = "Today is the First Thursday of the week"
text2 = "Jane needs to divide 30 pieces of candy equally among 6 kids. Each kid will get 5 pieces of candy."
apis = [wolframe_api]
generator = DataGenerator(config, model, tokenizer, apis=apis)
augumented_text_ids = generator.generate(text3)
#print(augumented_text_ids)
#print(tokenizer.decode(augumented_text_ids[0]))
if augumented_text_ids[0].size() != 0:
    print(tokenizer.decode(augumented_text_ids[0][0], skip_special_tokens=True))
else:
    print("loss less than threshold")