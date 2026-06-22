import ast
from numbers import Real

from llm.answer import get_answer


def _parse_saved_answer(raw_answer):
    """Parse old eval-cache lines without executing arbitrary code."""
    return ast.literal_eval(raw_answer)


def _is_score(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return 0.0 <= float(value) <= 1.0
    if isinstance(value, str):
        try:
            score = float(value)
        except ValueError:
            return False
        return 0.0 <= score <= 1.0
    return False


def _normalize_answer(llm_answer):
    if not isinstance(llm_answer, list):
        raise ValueError(f"LLM answer should be a list, got {type(llm_answer).__name__}: {llm_answer}")

    score_index = None
    for idx, item in enumerate(llm_answer):
        if _is_score(item):
            score_index = idx
            break
    if score_index is None:
        raise ValueError(f"Score answer is not correct: {llm_answer}")

    fusion_score = float(llm_answer[score_index])
    candidate_labels = [str(item).strip() for item in llm_answer[:score_index] if str(item).strip()]
    room_items = [str(item).strip() for item in llm_answer[score_index + 1 :] if str(item).strip()]

    if not room_items:
        raise ValueError(f"Room answer is not correct: {llm_answer}")

    # Some LLMs return several plausible rooms despite the prompt asking for one.
    # Treat that as the prompt's "everywhere" case so one broad answer will not
    # abort a long evaluation run.
    room = room_items[0] if len(room_items) == 1 else "everywhere"
    return candidate_labels, room, fusion_score

def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    label_existing = False

    with open(llm_answer_path, "a+") as f:
        f.seek(0)
        lines = f.readlines()

        for line in lines:
            if line.startswith(f"{label}:"):
                label_existing = True
                llm_answer = _parse_saved_answer(line[len(label) + 1 :].strip())
                print(f"Already have Answer for {label}: {llm_answer}")
                break

        if not label_existing:
            llm_answer, response = get_answer(prompt=label, client=llm_client)
            print(llm_answer)
            f.write(f"\n{label}: {llm_answer}")
            print(f"New Answer for {label}: {llm_answer}")
            # Write the response to the llm_response_path file
            with open(llm_response_path, "a+") as response_file:
                response_file.write(
                    f"\n{label}: {response}"
                )  # Write the label and its corresponding response to the file
                print(f"Response saved to {llm_response_path}: {response}")
    return _normalize_answer(llm_answer)
