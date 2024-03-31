import PyPDF2
from generate_classification_data.find_similar_page_of_query import get_category


def split_pdf(input_pdf, output_pdf, start_page, end_page):
    pdf_reader = PyPDF2.PdfReader(input_pdf)

    pdf_writer = PyPDF2.PdfWriter()

    for page_num in range(start_page - 1, end_page):
        pdf_writer.add_page(pdf_reader.pages[page_num])

    with open(output_pdf, 'wb') as output:
        pdf_writer.write(output)

    print(f"PDF已成功拆分为{start_page}至{end_page}页，并保存为{output_pdf}。")


def split_pdf_by_categories(input_pdf, output_folder):
    pdf_reader = PyPDF2.PdfReader(input_pdf)
    total_pages = len(pdf_reader.pages)

    category_ranges = [
        (1, 67),
        (68, 222),
        (223, 309),
        (310, 457),
        (458, 588),
        (589, 689),
        (690, 829),
        (830, 993),
        (994, 1162),
        (1163, 1228),
        (1229, 1594),
        (1595, 1706),
        (1707, 1941),
        (1942, 2123),
        (2124, 2338),
        (2339, 2395),
        (2396, 2579),
        (2580, 2809),
        (2810, 3209),
        (3210, 3295),
        (3296, 3314),
        (3315, 3494),
        (3495, total_pages)
    ]

    for start_page, end_page in category_ranges:
        category, idx = get_category(start_page)
        output_pdf = f"{output_folder}/{category}.pdf"
        split_pdf(input_pdf, output_pdf, start_page, end_page)


if __name__ == "__main__":
    split_pdf_by_categories("dataset/The Merck Manual of Diagnosis & Therapy, 19th Edition.pdf",
                            "dataset_per_chapter")
