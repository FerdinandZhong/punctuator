from inference import InferenceArguments, Inference
import pytest


testing_args = InferenceArguments(
    model_name_or_path="models/punctuator",
    tokenizer_name="distilbert-base-uncased",
    tag2id_storage_path="models/tag2id.json"
)

batch_test_text = [
    "how are you its been ten years since we met in shanghai im really happy to meet you again whats your current phone number",
    "my number is 82732212",
]

long_test_text = [
    "the two most likely largest inventions of our generation are the internet and the mobile phone   theyve changed the world   however   largely to our surprise   they also turned out to be the perfect tools for the surveillance state   it turned out that the capability to collect data   information and connections about basically any of us and all of us is exactly what weve been hearing throughout of the summer through revelations and leaks about western intelligence agencies   mostly u  s   intelligence agencies   watching over the rest of the world  weve heard about these starting with the revelations from june 6   edward snowden started leaking information   top secret classified information   from the u  s   intelligence agencies   and we started learning about things like prism and xkeyscore and others   and these are examples of the kinds of programs u  s   intelligence agencies are running right now   against the whole rest of the world  and if you look back about the forecasts on surveillance by george orwell   well it turns out that george orwell was an optimist     we are right now seeing a much larger scale of tracking of individual citizens than he could have ever imagined  and this here is the infamous nsa data center in utah   due to be opened very soon   it will be both a supercomputing center and a data storage center   you could basically imagine it has a large hall filled with hard drives storing data they are collecting   and its a pretty big building   how big   well   i can give you the numbers  140  000 square meters  but that doesnt really tell you very much   maybe its better to imagine it as a comparison   you think about the largest ikea store youve ever been in   this is five times larger   how many hard drives can you fit in an ikea store   right   its pretty big"
]

test_text_list = [batch_test_text, long_test_text]

@pytest.mark.parametrize("test_text", test_text_list)
def test_inference(test_text):
    inference = Inference(testing_args)
    results = inference.punctuation(test_text)
    assert len(results) == len(test_text)
    for result in results:
        assert result[0].isupper()
    inference.terminate()

