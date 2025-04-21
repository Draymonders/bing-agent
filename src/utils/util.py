from IPython.display import Image, display

def show_graph(img_path, graph):
    mermaid_png = graph.get_graph().draw_mermaid_png()
    with open(img_path, "wb") as f:
        f.write(mermaid_png)