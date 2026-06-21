import os
# So that the cache is not saved to disk C
#os.environ["HF_HOME"] = "D:\\huggingface_cache"


from datasets import load_dataset


def prepare_dataset_files(
        dataset_name: str,
        code_output_file: str,
        reference_output_file: str,
        num_samples: int = 20
):
    print(f"Загрузка датасета {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")

    print("Сортировка по quality_score...")
    sorted_dataset = dataset.sort("quality_score", reverse=True)
    top_samples = sorted_dataset.select(range(num_samples))

    print(f"Генерация файлов: {code_output_file} и {reference_output_file}...")

    with open(code_output_file, 'w', encoding='utf-8') as f_code, \
            open(reference_output_file, 'w', encoding='utf-8') as f_ref:

        f_ref.write("# Эталонная документация (Expected Output)\n\n---\n\n")

        for i, row in enumerate(top_samples):
            func_id = i + 1
            code_content = row.get('function_code', '').strip()
            doc_content = row.get('documentation', '').strip()
            lang = row.get('language', 'python').lower()

            if not code_content:
                continue

            f_code.write(f"### Функция ID: {func_id}\n\n")
            f_code.write("**Исходный код:**\n")
            f_code.write(f"```{lang}\n{code_content}\n```\n\n")
            f_code.write("\n\n")
            f_code.write("---\n\n")

            f_ref.write(f"### Функция ID: {func_id}\n\n")
            f_ref.write(f"{doc_content}\n\n")
            f_ref.write("---\n\n")

    print(f"Готово! Топ {num_samples} функций сохранены.")


if __name__ == "__main__":
    prepare_dataset_files(
        dataset_name="kaanrkaraman/code2doc",
        code_output_file="functions_for_llm_400.md",
        reference_output_file="reference_docs_400.md",
        num_samples=400
    )