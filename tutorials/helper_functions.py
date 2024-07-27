import numpy as np
import pandas as pd
import networkx as nx

from matscipy.neighbours import neighbour_list

from nglview import show_ase, ASEStructure

from bokeh.plotting import figure, from_networkx, show, save, output_file
from bokeh.palettes import Spectral4
from bokeh.models import Circle, NodesAndLinkedEdges, MultiLine, LinearColorMapper, CustomJS, HoverTool, TapTool
from bokeh.palettes import Viridis256
from bokeh.layouts import row


def get_cutoff(atoms, n_neightbours, cutoff=10.):
    """
    Function to get the distance of n_th neighbour using matscipy neighbour list.
    """
    from matscipy.neighbours import neighbour_list

    i, d = neighbour_list("id", atoms, cutoff=cutoff)
    first_atom_d = d[i == 0]
    first_atom_d.sort()
    return first_atom_d[n_neightbours]


def show_structure(structure, name=''):
    # create an empty canvas
    view = show_ase(structure, default_representation=False)
    view.remove_component(view[0])

    # add new component to visualise
    component  = view.add_component(ASEStructure(structure), default_representation=False, name=name)
    
    # set up the vew as just atomic spheres with unit cell
    scale=0.5
    component.add_spacefill()
    component.update_spacefill(radiusType='covalent',
                            radiusScale=scale)
    component.add_unitcell()

    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}   

    tooltip_js = """
                this.stage.mouseControls.add('hoverPick', (stage, pickingProxy) => {
                    let tooltip = this.stage.tooltip;
                    if(pickingProxy && pickingProxy.atom && !pickingProxy.bond){
                        let atom = pickingProxy.atom;
                        if (atom.structure.name.length > 0){
                            tooltip.innerText = atom.atomname + " atom index " + atom.index + ": " + atom.structure.name;
                        } else {
                            tooltip.innerText = atom.atomname + " atom index " + atom.index;
                        }
                    } else if (pickingProxy && pickingProxy.bond){
                        let bond = pickingProxy.bond;
                        if (bond.structure.name.length > 0){
                        tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond: " + bond.structure.name;
                        } else {
                            tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond";
                        }
                    } else if (pickingProxy && pickingProxy.unitcell){
                        tooltip.innerText = "Unit cell";
                    } else if (pickingProxy && pickingProxy.distance){
                        let distance = pickingProxy.distance;
                        tooltip.innerText = "Distance: " + distance.atom1.atomname + distance.atom1.index + "-" + distance.atom2.atomname + distance.atom2.index;
                    } 
                });
                """
    view._js(tooltip_js)
    return view


def interactive_neighbour_map(structure, 
                              target_atom_index=0,
                              target_img_size=32,
                              cutoff_factor=1.2):
    """Plots equivalent graph on the left and resulting image on the right.
       Hovering over nods shows the corresponding atom index and highlights
       all connected edges and clicking on it will show the weights (1/distance)
       for each edge and highlights the corresponding line on the image. 
       Hovering over the image displays the information for each pixel: 
       central atom index (row number), target atom index, distance and weight. 
       Clicking on a pixel  highlights the row and corresponding node on the graph on the left part. 
       To reset the selection click on the empty area of the plot.
       This might sound a but complicated, just start playing with the  plot and you will understand it very quick.

    Args:
        structure (ase.Atoms): Atoms to build the map for.
        target_atom_index (int, optional): Index of the central ato_. Defaults to 0.
        target_img_size (int, optional): Target image size. Defaults to 32.
        cutoff_factor (float, optional): Controls the number of neighbours 
                                         for the central atom taken into account. Defaults to 1.2.
    """    

    i, j, d = neighbour_list("ijd", structure, cutoff=cutoff_factor * get_cutoff(structure, target_img_size))

    sorting = np.argsort(d[i==target_atom_index])

    neighbour_indices = j[i==target_atom_index][sorting]
    img_size = min(len(neighbour_indices), target_img_size)
    
    source = i[i==target_atom_index][:img_size]

    target = neighbour_indices[:img_size]
    distance = d[i==target_atom_index][sorting][:img_size]

    sphere_indices = np.append(neighbour_indices, target_atom_index)

    for neighbour_index in neighbour_indices[:img_size - 1]:

        distances = structure.get_distances(neighbour_index, sphere_indices, mic=True)
        sorting = np.argsort(distances)
        # first element will be zero since this is the distance from the atom itself
        # thus we simply ignore the first record
        target = np.append(target, sphere_indices[sorting][1:img_size + 1])
        distance = np.append(distance, distances[sorting][1:img_size + 1])

        source = np.append(source, np.full(img_size, neighbour_index))

    weights = 1 / (distance).reshape((img_size, img_size))
    
    pd_edges = pd.DataFrame({"source": source,
                             "target": target,
                             "distance": distance,
                             "weight": weights.flatten()})
                         
    G = nx.from_pandas_edgelist(pd_edges, edge_attr=True)

    mapper = LinearColorMapper(palette=Viridis256)

    p = figure(x_range=(-2, 2), y_range=(-2, 2),
               x_axis_location=None, y_axis_location=None,
               tools="hover, tap", tooltips="index: @index")
    
    p.toolbar_location = None
    p.grid.grid_line_color = None

    graph = from_networkx(G, nx.spring_layout, scale=1.8, center=(0,0))

    p.renderers.append(graph)

    graph.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    #graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph.edge_renderer.selection_glyph = MultiLine(line_color={'field': 'weight', 'transform': mapper}, line_width=5)
    graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    # application of the attention factor for the image rows with 1/r^beta
    A = np.concatenate(([1], distance[:img_size - 1])) ** (-1./6.)
    image_weights = A.reshape((-1, 1)) / distance.reshape((img_size, img_size))

    image_weights = np.pad(image_weights, ((0, target_img_size - img_size), (0, target_img_size - img_size)))
    
    target = target.reshape(img_size, img_size)
    target = np.pad(target, ((0, target_img_size - img_size), (0, target_img_size - img_size)), constant_values=-1)
    
    distance = distance.reshape(img_size, img_size)
    distance = np.pad(distance, ((0, target_img_size - img_size), (0, target_img_size - img_size)))   
    

    data = {"weights": [np.array([image_weights[i,:]]) for i in range(target_img_size)],
            "target": [np.array([target[i,:]]) for i in range(target_img_size)],
            "distance": [np.array([distance[i,:]]) for i in range(target_img_size)],
            "x": np.zeros(target_img_size),
            "y": np.arange(target_img_size)[::-1],
            "dw": np.full(target_img_size, target_img_size),
            "dh": np.full(target_img_size, 1),
            "index": np.pad(source[::img_size], (0, target_img_size - img_size), constant_values=-1)}

    TOOLTIPS = [('index', "@index"),
                ('target', '@target'),
                ("weight", "@weights"),
                ('distance', '@distance')]

    img = figure(tools=())
    img.toolbar_location = None

    color_mapper = LinearColorMapper(palette="Viridis256", low=image_weights.min(), high=image_weights.max())

    img.image(source=data, image="weights", color_mapper=color_mapper)
    xx, yy = np.meshgrid(np.arange(target_img_size), np.arange(target_img_size))

    size = 0.825 * img.height / (target_img_size)

    img.square(xx.flatten() + 0.5, yy.flatten() + 0.5, size=size, line_width=2.,
               fill_color="grey", line_color = "firebrick", hover_line_alpha=1.0, alpha=0.0)


    img.add_tools(TapTool(renderers=[img.renderers[0]]))
    img.add_tools(HoverTool(renderers=[img.renderers[0]], tooltips=TOOLTIPS, point_policy='snap_to_data'))
    img.add_tools(HoverTool(renderers=[img.renderers[1]], tooltips=None, point_policy='snap_to_data'))

    nodes = graph.node_renderer.data_source
    image_source = img.renderers[0].data_source

    nodes_tap_handler = CustomJS(args=dict(nodes=nodes, image_source=image_source), code='''
                                const inds = nodes.selected.indices
                                console.log('Selection node at: ' + inds)
                                image_source.selected.indices = inds
                                ''')

    nodes.selected.js_on_change("indices", nodes_tap_handler)

    image_tap_handler = CustomJS(args=dict(nodes=nodes, image_source=image_source), code='''
                                const inds = image_source.selected.indices
                                console.log('Selection image at: ' + inds)
                                nodes.selected.indices = inds
                                ''')

    image_source.selected.js_on_change("indices", image_tap_handler)
    layout = row(p, img)

    show(layout)