
import re
import json
import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from trialmatch.prompt import build_messages, parse_output

PROJECT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = PROJECT / "outputs/qwen15b_lora/best_adapter"
OUTPUTS = PROJECT / "outputs"

st.set_page_config(
    page_title="TrialMatch",
    page_icon="🩺",
    layout="wide"
)

st.title("TrialMatch")
st.caption("AI-assisted clinical trial eligibility screening")

st.warning(
    "Educational screening prototype only. This application is not a clinical "
    "enrollment decision tool. Its predictions may be incorrect and must not "
    "be used to make medical decisions."
)

# ---------------- MODEL ----------------

@st.cache_resource(show_spinner=False)
def load_model(use_lora):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )

    if use_lora:
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {ADAPTER_PATH}"
            )
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    model.to(device)
    model.eval()

    return tokenizer, model, device


def prepare_note(raw_note):
    lines = [
        line.strip()
        for line in raw_note.splitlines()
        if line.strip()
    ]

    sentences = [
        re.sub(r"^\s*\d+\.\s*", "", line).strip()
        for line in lines
    ]
    sentences = [s for s in sentences if s]

    numbered_note = "\n".join(
        f"{i}. {sentence}"
        for i, sentence in enumerate(sentences)
    )

    return numbered_note, sentences


def predict(tokenizer, model, device, note, sentences, criterion, criterion_type):
    messages = build_messages(note, criterion_type, criterion)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    if inputs["input_ids"].shape[1] > 1984:
        raise ValueError(
            "The patient record and criterion exceed the supported input "
            "length. Please shorten the record."
        )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = output[0, inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    result = parse_output(
        raw_output,
        criterion_type,
        len(sentences)
    )

    result["raw_output"] = raw_output
    result["evidence"] = [
        {"id": i, "text": sentences[i]}
        for i in result["evidence_ids"]
    ]

    return result


# ---------------- PDF REPORT ----------------

def create_pdf(patient_note, model_name, results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak
    )

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["Normal"],
        fontSize=8,
        leading=11
    ))

    story = []

    story.append(Paragraph("TrialMatch Screening Report", styles["Title"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Educational screening prototype. Not a clinical enrollment "
        "decision or medical recommendation.</b>",
        styles["SmallNote"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Selected model:</b> {html.escape(model_name)}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Record", styles["Heading2"]))
    for i, line in enumerate(patient_note.splitlines()):
        if line.strip():
            story.append(Paragraph(
                f"<b>{i + 1}.</b> {html.escape(line)}",
                styles["SmallNote"]
            ))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Criterion-level Results", styles["Heading2"]))

    table_data = [[
        "Type", "Criterion", "Predicted status"
    ]]

    for item in results:
        table_data.append([
            Paragraph(html.escape(item["type"].title()), styles["SmallNote"]),
            Paragraph(html.escape(item["criterion"]), styles["SmallNote"]),
            Paragraph(
                html.escape(item["result"]["label"] or "Unable to determine"),
                styles["SmallNote"]
            )
        ])

    table = Table(
        table_data,
        colWidths=[25 * mm, 100 * mm, 45 * mm],
        repeatRows=1
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Supporting Evidence", styles["Heading2"]))

    for i, item in enumerate(results, 1):
        result = item["result"]

        story.append(Paragraph(
            f"<b>{i}. {html.escape(item['type'].title())}: "
            f"{html.escape(item['criterion'])}</b>",
            styles["Normal"]
        ))

        if result["evidence"]:
            for evidence in result["evidence"]:
                story.append(Paragraph(
                    f"Sentence {evidence['id']}: "
                    f"{html.escape(evidence['text'])}",
                    styles["SmallNote"]
                ))
        else:
            story.append(Paragraph(
                "No supporting sentence IDs returned.",
                styles["SmallNote"]
            ))

        if result["label"] is None:
            story.append(Paragraph(
                "Warning: no valid eligibility label was recovered.",
                styles["SmallNote"]
            ))

        story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Important: Predictions and evidence are generated by a machine "
        "learning model. They may be incomplete or incorrect. A missing "
        "evidence sentence does not establish that a criterion is unmet. "
        "All results require qualified human review.",
        styles["SmallNote"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- NAVIGATION ----------------

screen_tab, benchmark_tab = st.tabs([
    "Eligibility Screening",
    "Model Benchmarks"
])

with screen_tab:
    st.subheader("1. Patient record")
    st.caption(
        "Enter one sentence per line. Sentence IDs are assigned automatically "
        "and used to identify supporting evidence."
    )

    patient_note = st.text_area(
        "Patient information",
        height=190,
        placeholder=(
            "Patient is 58 years old.\n"
            "Diagnosed with type 2 diabetes in 2018.\n"
            "No history of active malignancy."
        ),
        key="patient_note_input"
    )

    st.subheader("2. Eligibility criteria")

    col1, col2 = st.columns(2)

    with col1:
        inclusion_text = st.text_area(
            "Inclusion criteria (one per line)",
            height=160,
            placeholder=(
                "Age between 18 and 65 years\n"
                "Confirmed diagnosis of type 2 diabetes"
            )
        )

    with col2:
        exclusion_text = st.text_area(
            "Exclusion criteria (one per line)",
            height=160,
            placeholder=(
                "History of active malignancy\n"
                "Severe renal impairment"
            )
        )

    st.subheader("3. Model selection")

    model_choice = st.radio(
        "Choose the Transformer",
        ["Fine-tuned Qwen (LoRA)", "Base Qwen"],
        horizontal=True
    )

    use_lora = model_choice == "Fine-tuned Qwen (LoRA)"

    st.caption(
        "Both models use greedy decoding with the same prompt format."
    )

    def get_criteria(text):
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    if st.button(
        "Run eligibility screening",
        type="primary",
        use_container_width=True
    ):
        inclusion = get_criteria(inclusion_text)
        exclusion = get_criteria(exclusion_text)

        if not patient_note.strip():
            st.error("Please enter a patient record.")
        elif not inclusion and not exclusion:
            st.error("Please enter at least one criterion.")
        else:
            note, sentences = prepare_note(patient_note)

            try:
                with st.spinner(
                    "Loading the selected model. First launch may take a while..."
                ):
                    tokenizer, model, device = load_model(use_lora)

                criteria = (
                    [("inclusion", c) for c in inclusion]
                    + [("exclusion", c) for c in exclusion]
                )

                results = []
                progress = st.progress(0)

                for index, (criterion_type, criterion) in enumerate(criteria):
                    result = predict(
                        tokenizer,
                        model,
                        device,
                        note,
                        sentences,
                        criterion,
                        criterion_type
                    )

                    results.append({
                        "type": criterion_type,
                        "criterion": criterion,
                        "result": result
                    })

                    progress.progress((index + 1) / len(criteria))

                st.session_state["screening_results"] = results
                st.session_state["screening_model"] = model_choice
                st.session_state["screening_patient"] = patient_note

                st.success("Screening completed.")

            except Exception as error:
                st.error(f"Screening failed: {error}")

    if "screening_results" in st.session_state:
        results = st.session_state["screening_results"]

        st.divider()
        st.subheader("4. Criterion-level results")

        st.caption(
            f"Selected model: {st.session_state['screening_model']}"
        )

        for index, item in enumerate(results, 1):
            result = item["result"]
            label = result["label"] or "Unable to determine"

            with st.expander(
                f"{index}. {item['type'].title()} | {label}",
                expanded=True
            ):
                st.markdown(f"**Criterion:** {item['criterion']}")
                st.markdown(f"**Predicted status:** {label}")

                st.markdown("**Supporting evidence**")

                if result["evidence"]:
                    for evidence in result["evidence"]:
                        st.markdown(
                            f"- **Sentence {evidence['id']}:** "
                            f"{evidence['text']}"
                        )
                else:
                    st.info("No supporting sentence IDs returned.")

                if result["label"] is None:
                    st.warning(
                        "No valid eligibility label could be recovered."
                    )

                if not result["valid_json"]:
                    st.warning(
                        "The model response was not valid JSON."
                    )

        st.divider()
        st.subheader("5. Export screening report")

        pdf = create_pdf(
            st.session_state["screening_patient"],
            st.session_state["screening_model"],
            results
        )

        st.download_button(
            "Download screening report (PDF)",
            data=pdf,
            file_name="trialmatch_screening_report.pdf",
            mime="application/pdf",
            type="primary"
        )

        csv_rows = []
        for item in results:
            result = item["result"]
            csv_rows.append({
                "criterion_type": item["type"],
                "criterion": item["criterion"],
                "predicted_label": result["label"],
                "evidence_ids": result["evidence_ids"],
                "evidence_text": " | ".join(
                    e["text"] for e in result["evidence"]
                ),
                "valid_json": result["valid_json"]
            })

        csv_data = pd.DataFrame(csv_rows).to_csv(index=False)

        st.download_button(
            "Download results (CSV)",
            data=csv_data,
            file_name="trialmatch_screening_results.csv",
            mime="text/csv"
        )


# ---------------- BENCHMARK DASHBOARD ----------------

with benchmark_tab:
    st.subheader("Decoding and attention benchmarks")

    st.caption(
        "Experimental results recorded during Tasks 3, 4 and 5. "
        "Measurements were collected on a Tesla T4 GPU."
    )

    task3_file = OUTPUTS / "task3/kv_cache_benchmark.csv"
    task4_file = OUTPUTS / "task4/attention_benchmark.csv"
    task5_file = OUTPUTS / "task5/speculative_benchmark.csv"

    # KV CACHE
    st.markdown("### Task 3: KV-cache decoding")

    if task3_file.exists():
        kv = pd.read_csv(task3_file)

        st.markdown(
            "Compares ordinary greedy decoding, which recomputes the "
            "sequence, against greedy decoding with a KV cache."
        )

        kv["speedup"] = (
            kv["reference_total_s"] / kv["cached_total_s"]
        )
        kv["comparison"] = (
            kv["input_tokens"].astype(str)
            + " input / "
            + kv["output_tokens"].astype(str)
            + " output"
        )

        st.dataframe(
            kv[[
                "input_tokens",
                "output_tokens",
                "reference_total_s",
                "cached_total_s",
                "speedup",
                "token_agreement"
            ]].rename(columns={
                "reference_total_s": "Ordinary (s)",
                "cached_total_s": "KV-cache (s)",
                "speedup": "Speedup (x)",
                "token_agreement": "Exact agreement"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            kv.set_index("comparison")[[
                "reference_total_s",
                "cached_total_s"
            ]].rename(columns={
                "reference_total_s": "Ordinary greedy",
                "cached_total_s": "KV-cache"
            })
        )

        mean_speedup = kv["reference_total_s"].sum() / kv["cached_total_s"].sum()
        st.metric("Aggregate time ratio", f"{mean_speedup:.2f}x")

        if kv["token_agreement"].astype(str).str.lower().isin(
            ["true", "1"]
        ).all():
            st.success("Exact token agreement in all recorded comparisons.")

    else:
        st.warning("Task 3 benchmark file not found.")

    st.divider()

    # ATTENTION
    st.markdown("### Task 4: Attention implementation")

    if task4_file.exists():
        attn = pd.read_csv(task4_file)

        st.markdown(
            "Compares reference attention, tiled online attention and "
            "PyTorch scaled dot-product attention (SDPA)."
        )

        st.dataframe(
            attn.rename(columns={
                "sequence_length": "Sequence length",
                "method": "Method",
                "median_time_seconds": "Median time (s)",
                "peak_extra_memory_GB": "Peak extra memory (GB)"
            }),
            use_container_width=True,
            hide_index=True
        )

        time_chart = attn.pivot(
            index="sequence_length",
            columns="method",
            values="median_time_seconds"
        )
        st.markdown("**Median execution time**")
        st.line_chart(time_chart)

        memory_chart = attn.pivot(
            index="sequence_length",
            columns="method",
            values="peak_extra_memory_GB"
        )
        st.markdown("**Peak extra memory**")
        st.line_chart(memory_chart)

        st.info(
            "The tiled online implementation reduces intermediate memory "
            "usage compared with the reference implementation in these tests, "
            "but its Python-loop implementation is slower. PyTorch SDPA "
            "performed best in the recorded benchmarks."
        )

    else:
        st.warning("Task 4 benchmark file not found.")

    st.divider()

    # SPECULATIVE
    st.markdown("### Task 5: Speculative decoding")

    summary_file = OUTPUTS / "task5/speculative_benchmark_summary.json"

    if task5_file.exists() and summary_file.exists():
        spec = pd.read_csv(task5_file)
        summary = json.loads(summary_file.read_text())

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Ordinary latency",
            f"{summary['mean_ordinary_latency_sec']:.2f} s"
        )
        c2.metric(
            "Speculative latency",
            f"{summary['mean_speculative_latency_sec']:.2f} s"
        )
        c3.metric(
            "Mean acceptance rate",
            f"{summary['mean_acceptance_rate']:.1%}"
        )

        st.dataframe(
            spec[[
                "input_tokens",
                "output_tokens",
                "ordinary_latency_sec",
                "speculative_latency_sec",
                "speedup",
                "acceptance_rate",
                "token_agreement"
            ]].rename(columns={
                "input_tokens": "Input tokens",
                "output_tokens": "Output tokens",
                "ordinary_latency_sec": "Ordinary (s)",
                "speculative_latency_sec": "Speculative (s)",
                "speedup": "Speedup (x)",
                "acceptance_rate": "Acceptance rate",
                "token_agreement": "Exact agreement"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            spec.set_index("prompt")[[
                "ordinary_latency_sec",
                "speculative_latency_sec"
            ]].rename(columns={
                "ordinary_latency_sec": "Ordinary greedy",
                "speculative_latency_sec": "Speculative"
            })
        )

        st.warning(
            "Speculative decoding was slower in these experiments: mean "
            f"speedup was {summary['mean_speedup']:.2f}x. The implementation "
            "uses sequential target verification to preserve exact greedy "
            "token agreement under the tested BF16 setup. Draft-model "
            "overhead therefore outweighed the benefit of drafting."
        )

        st.caption(
            f"Draft model: {summary['draft_model']} | "
            f"Draft length: {summary['draft_k']} | "
            f"Accepted {summary['total_accepted_tokens']} of "
            f"{summary['total_drafted_tokens']} drafted tokens."
        )

        if spec["token_agreement"].astype(str).str.lower().isin(
            ["true", "1"]
        ).all():
            st.success("Exact token agreement in all recorded comparisons.")

    else:
        st.warning("Task 5 benchmark files not found.")

    st.divider()

    st.caption(
        "Benchmark results are from the recorded experimental setup and "
        "should not be interpreted as universal performance guarantees."
    )
