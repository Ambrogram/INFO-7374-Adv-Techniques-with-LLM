from src.prompt_utils import prepend_instruction
from src.constants import INSTR


def test_prepend_instruction_guards_double():
    original = INSTR + "Fever and cough for 3 days."
    out = prepend_instruction(original)
    assert out == original


def test_prepend_instruction_when_empty():
    out = prepend_instruction("")
    assert out.startswith("Suppose that you are a medical diagnosis assistant")


def test_prepend_instruction_normal_case():
    msg = "Fever and cough for 3 days."
    out = prepend_instruction(msg)
    assert out.startswith("Suppose that you are a medical diagnosis assistant")
    assert out.endswith(msg)


