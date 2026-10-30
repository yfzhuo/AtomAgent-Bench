TOOLS = [
    {"type": "function", "function": {
        "name": "view",
        "description": (
            "Render the structure to an image to inspect it visually. You may view "
            "the CURRENT (edited) structure or the INPUT structure, from a chosen "
            "viewpoint, optionally with atom index labels and/or highlighted atoms. "
            "You can NEVER view the target/answer structure."),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "enum": ["current", "input"],
                       "description": "Which structure to look at. Default 'current'."},
            "axis": {"type": "string", "enum": ["a", "b", "c"],
                     "description": "Look straight down a lattice axis (optional)."},            "azim": {"type": "number",

                     "description": "Custom azimuth (optional, overrides axis)."},
            "elev": {"type": "number", "description": "Custom elevation (optional)."},
            "show_indices": {"type": "boolean",
                             "description": "Label each atom with its integer index."},
            "highlight": {"type": "array", "items": {"type": "integer"},
                          "description": "Atom indices to mark. Each highlighted "
                           "atom is circled in red: a SOLID red outline means the "
                           "atom is visible (in front); a DASHED red outline means "
                           "the atom is occluded (hidden behind nearer atoms). Use "
                           "this to point at the atoms you intend to operate on."}},
            "required": []}}},

    {"type": "function", "function": {
        "name": "run_python",
        "description": (
            "Execute Python code to transform the structure. A variable `structure` "
            "(pymatgen.core.Structure, a copy of the current state) is preloaded; "
            "`input_cif` (str) and the classes Structure/Element/Lattice plus numpy "
            "as np are available. Modify `structure` in place or reassign it; whatever "
            "`structure` holds after your code runs becomes the new current structure. "
            "Use print() to inspect values. Write whatever logic the task needs."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"]}}},

    {"type": "function", "function": {
        "name": "submit",
        "description": "Finish: export the current structure as CIF and submit it "
                       "for scoring. Call when confident it is correct.",
        "parameters": {"type": "object", "properties": {}}}},
]
