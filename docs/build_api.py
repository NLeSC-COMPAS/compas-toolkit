def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(name, items):
    content = rst_comment()
    content += f".. _{name}:\n\n"
    content += name + "\n" + "=" * len(name) + "\n"

    for item in items:
        directive = "doxygenclass" if item[0].isupper() else "doxygenfunction"
        content += f".. {directive}:: kmm::{item}\n"

    filename = f"api_cxx/{name}.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


def build_index_page(groups):
    body = ""
    children = []

    for groupname, symbols in groups.items():
        body += f".. raw:: html\n\n   <h2>{groupname}</h2>\n\n"

        for symbol in symbols:
            if isinstance(symbol, str):
                name = symbol
                items = [symbol]
            else:
                name, items = symbol

            filename = build_doxygen_page(name, items)
            children.append(filename)

            filename = filename.replace(".rst", "")
            body += f"* :doc:`{name} <{filename}>`\n"

        body += "\n"

    title = "C++ API Reference"
    content = rst_comment()
    content += title + "\n" + "=" * len(title) + "\n"
    content += ".. toctree::\n"
    content += "   :titlesonly:\n"
    content += "   :hidden:\n\n"

    for filename in sorted(children):
        content += f"   {filename}\n"

    content += "\n"
    content += body + "\n"

    filename = "api_cxx.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


groups = {
}


build_index_page(groups)
