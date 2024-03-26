def extract_QA_pairs(file_path):
    read_question = False
    read_answer = False
    read_source = True
    current_question, current_answer = "", ""
    res = []
    with open(file_path, "r") as f:
        text_list = f.readlines()
    for text in text_list:
        text = text.strip()
        if text.startswith("Question: "):
            idx = len('Question: ')
            current_question = text[idx:]

            read_question = True
            read_source = False
            read_answer = False
        elif text.startswith("Answer: "):

            idx = len("Answer: ")
            current_answer = text[idx:]

            read_answer = True
            read_source = False
            read_question = False
        elif text.startswith("Sources:"):

            res.append([current_question, current_answer])
            read_source = True
            read_answer = False
            read_question = False
        else:
            if read_question:
                current_question += " " + text.strip()
            if read_answer:
                current_answer += " " + text.strip()

            if read_source: continue

    for query, val in res:
        print(query)
        print(val)

        print("\n\n")


extract_QA_pairs("outputs/answers.txt")
