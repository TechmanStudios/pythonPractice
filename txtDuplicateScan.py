def remove_duplicate_sections(input_file, output_file):
    """
    Reads a text file, identifies sections that begin with '###',
    and writes only the unique sections (first occurrence) to a new file.
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_section_lines = []

    # We'll track sections in a set to detect duplicates.
    seen_sections = set()

    for line in lines:
        # If a line starts with '###', we know we're beginning a *new* section.
        if line.startswith('###'):
            # If we already collected lines for a previous section, 
            # wrap it up and add to the sections list.
            if current_section_lines:
                section_text = "".join(current_section_lines)
                sections.append(section_text)
                current_section_lines = []
        
        # Add the current line to our in-progress section buffer.
        current_section_lines.append(line)

    # After the loop, make sure to add the last section if there's anything left in the buffer.
    if current_section_lines:
        section_text = "".join(current_section_lines)
        sections.append(section_text)

    # Now we have a list of all sections. Let's weed out duplicates.
    unique_sections = []
    for section in sections:
        # If we haven’t seen this exact block of text, we keep it.
        if section not in seen_sections:
            seen_sections.add(section)
            unique_sections.append(section)
        # If we *have* seen it, we do nothing—effectively removing the duplicate.

    # Write only unique sections to the output file.
    with open(output_file, 'w', encoding='utf-8') as f:
        for section in unique_sections:
            f.write(section)


if __name__ == "__main__":
    input_file_path = r"G:\GPTs\QLI\vimeoText.txt"
    output_file_path = r"G:\GPTs\QLI\vimeoText_draft1.txt"

    remove_duplicate_sections(input_file_path, output_file_path)
    print("Done! A brand-new file without duplicate sections is ready.")
