def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(name, items):
    content = rst_comment()
    content += f".. _{name}:\n\n"
    content += name + "\n" + "=" * len(name) + "\n"

    for item in items:
        is_struct = any(c.lower() != c for c in item) and '(' not in item

        directive = "doxygenstruct" if is_struct else "doxygenfunction"
        content += f".. {directive}:: compas::{item}\n"

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
    "Data structures": [
        "CudaContext",
        #"TissueParameterField",
        "TissueParameters",
    ],
    "Sequences": [
        "FISPSequence",
        "pSSFPSequence",
        ("simulate_magnetization",[
        "simulate_magnetization(const CudaContext&, TissueParameters, FISPSequence)",
        "simulate_magnetization(const CudaContext&, TissueParameters, pSSFPSequence)",
        ]),
        ("simulate_magnetization_derivative",[
            "simulate_magnetization_derivative(const CudaContext &, int, Array<cfloat, 2>, TissueParameters, FISPSequence, float)",
            "simulate_magnetization_derivative(const CudaContext &, int, Array<cfloat, 2>, TissueParameters, pSSFPSequence, float)"
        ]),
    ],
    "Trajectories": [
        "Trajectory",
        "CartesianTrajectory",
        "SpiralTrajectory",
        "SimulateSignalMethod",
        "magnetization_to_signal",
        "phase_encoding",
    ],
    "Jacobian computation": [
        "compute_jacobian",
        "compute_jacobian_hermitian",
    ],
    "Utilities": [
        "compute_residual",
    ],
}


build_index_page(groups)
