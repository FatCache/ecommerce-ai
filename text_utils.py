
def extract_yaml_from_markdown(text):
    """Extracts YAML content from a markdown code block."""
    if text.strip().startswith('```yaml'):
        # Find the start and end of the YAML block
        start_index = text.find('```yaml') + len('```yaml')
        end_index = text.rfind('```')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index:end_index].strip()
    return text.strip()