class PromptAssembler:
    def __init__(self, template_path):
        """
        Initialize the PromptAssembler with the path to the prompt template.

        Args:
            template_path (str): Path to the .txt file containing the prompt template.
        """
        self.template_path = template_path
        self.template = self._load_template()

    def _load_template(self):
        """
        Load the prompt template from the specified file.

        Returns:
            str: The content of the template file.
        """
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found at {self.template_path}")

    def assemble_prompt(self, query, retrieved_docs, parsed_pdf):
        """
        Assemble the prompt for text generation.

        Args:
            query (str): The user query.
            retrieved_docs (list of str): Full texts of the retrieved documents.
            parsed_pdf (str): Full text from the parsed PDF.

        Returns:
            str: The assembled prompt.
        """
        # Combine the retrieved documents into a single string
        retrieved_docs_text = "\n\n".join(retrieved_docs)

        # Assemble the prompt using the template
        assembled_prompt = self.template.format(
            query=query,
            retrieved_docs=retrieved_docs_text,
            parsed_pdf=parsed_pdf
        )

        return assembled_prompt