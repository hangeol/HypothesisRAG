"""Stage-wise batched inference in separate GPU worker processes."""
import os
import pickle
import tempfile
import multiprocessing as _mp
from typing import List, Dict


def _format_chat_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Format chat messages into a prompt for vLLM generate()."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)


def _extract_vllm_output_texts(outputs) -> List[str]:
    """
    Normalize vLLM outputs (chat() or generate()) into a list of strings.
    Supports a few return shapes across vLLM versions.
    """
    if outputs is None:
        return []

    if isinstance(outputs, tuple):
        outputs = list(outputs)
    elif not isinstance(outputs, list):
        outputs = [outputs]

    texts = []
    for out in outputs:
        text = ""
        if hasattr(out, "outputs"):
            try:
                first = out.outputs[0]
                if hasattr(first, "text"):
                    text = first.text
            except Exception:
                text = ""
        elif isinstance(out, dict):
            if "choices" in out and out["choices"]:
                choice = out["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        text = message.get("content", "") or text
                    else:
                        text = choice.get("text", text)
                else:
                    text = getattr(choice, "text", text)
            elif "text" in out:
                text = out.get("text", "")
        elif isinstance(out, str):
            text = out

        texts.append("" if text is None else str(text))

    return texts


def _run_vllm_batch_phase(
    model_path: str,
    messages_list_file: str,
    output_file: str,
    gpu_mem: float,
    max_model_len: int,
    max_tokens: int,
    tensor_parallel_size: int = 1,
):
    """
    Subprocess entry: load vLLM model, run batch chat, save outputs, exit.
    All GPU memory is freed when this process exits.
    """
    from vllm import LLM, SamplingParams

    # Load messages
    with open(messages_list_file, "rb") as f:
        messages_list = pickle.load(f)

    print(f"    [subprocess] Loading {model_path} ...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    print(
        f"    [subprocess] Running batch inference on {len(messages_list)} inputs ..."
    )

    prompts = [_format_chat_messages(llm.get_tokenizer(), m) for m in messages_list]
    try:
        outputs = llm.chat(
            messages=messages_list,
            sampling_params=sampling,
            use_tqdm=True,
        )
    except Exception:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling,
            use_tqdm=True,
        )

    # Extract text outputs
    result_texts = _extract_vllm_output_texts(outputs)
    if len(result_texts) != len(messages_list):
        raise RuntimeError(
            f"Unexpected vLLM output count: got {len(result_texts)} for {len(messages_list)} inputs"
        )

    with open(output_file, "wb") as f:
        pickle.dump(result_texts, f)

    print(f"    [subprocess] Done. Exiting to free GPU memory.")


def _run_phase_in_subprocess(
    model_path: str,
    messages_list: list,
    gpu_mem: float,
    max_model_len: int,
    max_tokens: int,
    tensor_parallel_size: int = 1,
) -> list:
    """Run a vLLM batch in a subprocess and return result texts."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in:
        pickle.dump(messages_list, f_in)
        input_path = f_in.name

    output_path = input_path + ".out.pkl"

    try:
        p = _mp.get_context("spawn").Process(
            target=_run_vllm_batch_phase,
            args=(
                model_path,
                input_path,
                output_path,
                gpu_mem,
                max_model_len,
                max_tokens,
                tensor_parallel_size,
            ),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"vLLM subprocess failed (exit={p.exitcode})")

        with open(output_path, "rb") as f:
            return pickle.load(f)
    finally:
        for fp in [input_path, output_path]:
            try:
                os.unlink(fp)
            except OSError:
                pass
