import xml.etree.ElementTree as ET
import os


def xml_parser(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    res = []
    questions = root.findall(".//Question")
    for question in questions:
        res.append(question.text)
    return res


def get_all_files_path(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def parser_all_questions():
    all_questions = []
    for file_path in get_all_files_path("MedQuAD-master"):
        # print(file_path)
        all_questions.extend(xml_parser(file_path))

    if not os.path.exists("generate_classification_data/all_query.txt"):
        with open("generate_classification_data/all_query.txt", "w") as f:
            for question in all_questions:
                f.writelines(question + '\n')
    return all_questions


if __name__ == "__main__":
    all_questions = parser_all_questions()
    print(len(all_questions))
