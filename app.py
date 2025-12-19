# -------------------------------
# CONTEXTUAL CHUNK FILTER
# -------------------------------
def get_relevant_chunks(chunks, topic):
    good = []
    for ch in chunks:
        if topic.lower() in ch.lower() and 30 <= len(ch.split()) <= 140:
            if not any(x in ch.lower() for x in ["activity", "exercise", "project", "table", "figure"]):
                good.append(ch)
    return good


# -------------------------------
# QUESTION CAPACITY CHECK
# -------------------------------
def count_possible_mcqs(chunks):
    count = 0
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if sents:
            count += 1   # one good MCQ per chunk
    return count


# -------------------------------
# UPDATED NCERT MCQs
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()

    for ch in chunks:
        sentences = [s.strip() for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]

        if not sentences:
            continue

        correct = sentences[0]

        if correct in used:
            continue

        distractors = []
        for other in chunks:
            if other == ch:
                continue
            for s in re.split(r'[.;]', other):
                s = s.strip()
                if (
                    len(s.split()) >= 10
                    and topic.lower() in s.lower()
                    and s != correct
                ):
                    distractors.append(s)

        if len(distractors) < 3:
            continue

        used.add(correct)

        options = [highlight_keywords(correct)] + distractors[:3]
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(highlight_keywords(correct))
        })

        if len(mcqs) >= n:
            break

    return mcqs


# -------------------------------
# LOAD & PROCESS CONTENT
# -------------------------------
texts = load_all_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunks(t))

relevant_chunks = get_relevant_chunks(chunks, topic)

max_possible = count_possible_mcqs(relevant_chunks)

# -------------------------------
# SUBJECTIVE TAB (UNCHANGED)
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        final_n = min(num_q, max_possible)

        if max_possible == 0:
            st.error("No meaningful content found for this topic in NCERT.")
        else:
            st.info(f"📊 {max_possible} meaningful questions possible. Showing {final_n}.")
            qs = generate_subjective(topic, final_n)
            for i, q in enumerate(qs, 1):
                st.write(f"{i}. {q}")


# -------------------------------
# MCQs TAB (VALIDATED)
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"]
    )

    if st.button("Generate MCQs"):

        if max_possible == 0:
            st.error("❌ No meaningful NCERT content found for this topic.")
            st.stop()

        final_n = min(num_q, max_possible)
        st.info(f"📊 {max_possible} meaningful MCQs possible. Showing {final_n}.")

        if mcq_type == "NCERT MCQs":
            mcqs = generate_ncert_mcqs(relevant_chunks, topic, final_n)

            if not mcqs:
                st.warning("Not enough clean sentences to generate MCQs.")
            else:
                for i, m in enumerate(mcqs, 1):
                    st.write(f"**Q{i}. {m['q']}**")
                    for j, opt in enumerate(m["options"]):
                        st.write(f"{chr(97+j)}) {opt}")
                    st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                    st.write("---")

        elif mcq_type == "UPSC PYQ – Statements":
            qs = generate_upsc_statements(topic, final_n)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                for idx, s in enumerate(q["statements"], 1):
                    st.write(f"{idx}. {s}")
                st.write("Which of the statements given above are correct?")
                st.write(f"✅ Answer: {q['answer']}")
                st.write("---")

        else:
            qs = generate_assertion_reason(topic, final_n)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                st.write(f"**Reason (R):** {q['R']}")
                st.write("a) Both A and R are true and R is the correct explanation of A")
                st.write("b) Both A and R are true but R is not the correct explanation of A")
                st.write("c) A is true but R is false")
                st.write("d) A is false but R is true")
                st.write("✅ Answer: a")
                st.write("---")
