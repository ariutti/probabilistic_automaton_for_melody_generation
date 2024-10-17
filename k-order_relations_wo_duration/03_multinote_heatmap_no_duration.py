# This is a script to:
# + parse a midifile;
# + compute some statistics about notes/rest relationships;
# + build some graphs in order for these relationships to be easily visualized;
# + export these relationships in YAML format to be later precessed by other softwares;

# DEV NOTEs: the script doesn't keep track of notes/rests duration,
# it only uses notes pitches. Further improvements will be to consider also
# duration on notes/rests and also the music dynamics (MIDI velocity)
# and maybe more.

from music21 import converter, pitch, note, key
from graphviz import Digraph
import numpy as np

# Put here the name of the MIDI file you want to examine
MIDIFILENAME = "old_mc_donald.MID"
MIDIFILES_FOLDER = "../"
OUTPUT_FOLDER = "renders"

# some settings you may want to change in order for the script to behave diffently
# increase the 'MAX_ORDER' if you want to make more in-depth relationship
# calculation about the notes/rests sequences
MAX_ORDER = 5

# Set those booleans to save also the corresponding graph images
SAVE_SCORE = True
SHOW_PLOT  = False
SAVE_PLOT  = True
SAVE_DOT   = True
SAVE_YAML  = True

# DON'T TOUCH BELOW THIS LINE ++++++++++++++++++++++++++++++++++++++++++++++++++
# utilities and support variables ++++++++++++++++++++++++++++++++++++++++++++++
# a list of the notes and rest used inside tha piece
USED_NOTES_LIST = []
USED_MIDINOTES_LIST = []
MIN_MIDINOTE = 0
MAX_MIDINOTE = 0
MIDINOTE_RANGE = 0

# utilities and support functions ++++++++++++++++++++++++++++++++++++++++++++++
def save_musical_score( score, cleanUp=True ):
    import os, subprocess
    # Salva il file in formato LilyPond
    lilypond_file_path = f"{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}.ly"
    score.write('lily', lilypond_file_path)

    # Converti il file LilyPond in PNG utilizzando LilyPond
    #png_file_path = os.path.splitext(lilypond_file_path)[0] + ".png"
    subprocess.run(["lilypond", "-f", "png", "-o", f"{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}_score", lilypond_file_path])

    if cleanUp:
        os.chdir( OUTPUT_FOLDER )
        LY_FILES_LIST = [f for f in os.scandir( os.getcwd()) if ( f.is_file() and f.name.lower().endswith(".ly") ) ]
        EPS_FILES_LIST = [f for f in os.scandir( os.getcwd()) if ( f.is_file() and f.name.lower().endswith(".eps") ) ]
        TEX_FILES_LIST = [f for f in os.scandir( os.getcwd()) if ( f.is_file() and f.name.lower().endswith(".tex") ) ]
        TEXI_FILES_LIST = [f for f in os.scandir( os.getcwd()) if ( f.is_file() and f.name.lower().endswith(".texi") ) ]
        COUNT_FILES_LIST = [f for f in os.scandir( os.getcwd()) if ( f.is_file() and f.name.lower().endswith(".count") ) ]

        TO_BE_REMOVED_FILES = LY_FILES_LIST + EPS_FILES_LIST + TEX_FILES_LIST + TEXI_FILES_LIST + COUNT_FILES_LIST

        for toBeRemovedFile in TO_BE_REMOVED_FILES:
            os.remove( toBeRemovedFile )

# a function to save the finite state machine
# (fsm) as a YAML file in order for this fsm
# to be later interpreted by SuperCollider
def save_yaml( dictionary, _DEPTH ):
    import yaml
    # Path to YAML file
    yaml_file_path = f'{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}_{_DEPTH}_order_fsm.yaml'

    fsm_data = {}

    # general parameters
    #fsm_data[ "num_relationships"] = _DEPTH

    # in base of the number of elements, the dictionary keys could be:
    # NUM_ELEMENTS = 1 --> keys are of type 'string'
    # NUM_ELEMENTS >= 2 --> keys are of type 'tuple'
    for k in sorted(dictionary.keys()):

        if _DEPTH == 1:
            key = "undefined"
            dest = str(k)
            value = dictionary[ k ]
            #print(f"{key} --> {value}")

            #fsm_data[ key ] = value

            if key not in fsm_data:
               fsm_data[ key ] = []
            fsm_data[ key ].append( dest )
            fsm_data[ key ].append( value )

        elif _DEPTH >= 2:
            tuple_list = [str(e) for e in list(k)]
            key = "_".join( tuple_list[:-1] )
            dest = tuple_list[-1]
            value = dictionary[ k ]
            #print(f"{key} --> {dest} : {value}")
            #print( tuple_list )

            if key not in fsm_data:
                fsm_data[ key ] = []
            fsm_data[ key ].append( dest )
            fsm_data[ key ].append( value )

    # Write data on YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(fsm_data, yaml_file)

def get_list_of_used_notes( _score ):
    used_notes = set()
    # Scan all notes and rests in the MIDI file and
    # save all possible notes that exist in the song
    for i, element in enumerate(_score.recurse(classFilter=('Note', 'Rest'))):
        # save the current note inside the set only if it has not been save previously
        if isinstance(element, note.Note):
            #print("this is a note")
            if element.nameWithOctave not in used_notes:
                used_notes.add( element.nameWithOctave )
        else:
            #print( "this is a Rest")
            if "Rest" not in used_notes:
                used_notes.add( "Rest" )

    return used_notes

def get_pitchClassIndex_from_nomeName( noteNameString ):
    pitch_class = pitch.Pitch(noteNameString)
    pitch_class_index = pitch_class.pitchClass
    return pitch_class_index

def get_midinote_from_noteNameWithOctave( noteNameWithOctaveString ):
    # Create a Pitch object
    a = pitch.Pitch(noteNameWithOctaveString)
    # Get the MIDI note value
    midi_pitch = a.midi
    return midi_pitch

def make2Dgraph( score, _DEPTH ):

    # GET NOTES RELATIONSHIPS ******************************************************
    # Dictionary for keeping track of the relationship between notes and rests
    state_relationships = {}
    tmp_elements_sequence = []


    for i, element in enumerate(score.recurse(classFilter=('Note', 'Rest'))):
        """
        if( isinstance(element, note.Note) ):
            print( f"pitchClass: {element.pitch.pitchClass}" )
            print( f"accidental: {element.pitch.accidental}")
            print( f"name: {element.name}" )
            print( f"octave: {element.octave}" )
            print( f"nameWithOctave: {element.nameWithOctave}" )
            print( f"midi: {element.pitch.midi}" )
        else:
            print( "\nthis is a rest\n")

        #print( f"type: {element.pitch.type}" )
        print( f"duration (type): {element.duration.type}" )
        print( f"duration (quarter length): {element.quarterLength}")
        print()
        """

        # if the element is the first, second or otherwise less than _DEPTH, add it to the list
        if( i < _DEPTH):
            tmp_elements_sequence.append(element)
        else:
            # if the element index is greater than the width of the list
            # scale the elements and add to the list
            tmp_elements_sequence.pop(0)
            tmp_elements_sequence.append(element)

        # build or update existing relationship if hte element sequence has length == _DEPTH
        if( len(tmp_elements_sequence) == _DEPTH):
            l = [ n.nameWithOctave if isinstance(n, note.Note) else 'Rest' for n in tmp_elements_sequence]
            #print( l )
            relationship_key = tuple(l)
            #print( relationship_key )

            if relationship_key not in state_relationships:
                state_relationships[relationship_key] = 1
            else:
                state_relationships[relationship_key] += 1

    #print( used_notes )

    #print("\nSTATE RELATIONSHIPS")
    #print(f"We have {len(state_relationships)} relationships")
    #for key in sorted(state_relationships.keys()):
    #    print( f"{key} - {state_relationships[ key ]}" )

    # build a (sorted) list for every axis
    x_axis_list = sorted( USED_NOTES_LIST )

    #print("ASSE X")
    #print( x_axis_list )
    #print( len(x_axis_list) )

    y_axis_list = []
    for e in list(state_relationships.keys()):
        first_elements_in_relationship = e[:-1]
        #print( first_elements_in_relationship )
        # put on y axis only non redundant elements
        if first_elements_in_relationship not in y_axis_list:
            y_axis_list.append( first_elements_in_relationship )
    y_axis_list = sorted(y_axis_list)

    #print("ASSE Y")
    #print( y_axis_list )
    #print( len(y_axis_list) )


    # Construct an empty matrix with the appropriate dimensions
    matrix = np.zeros((len(y_axis_list), len(x_axis_list)))
    # Filling the matrix with relationship counts
    for relationship, count in state_relationships.items():
        start = tuple( [ n for n in relationship[:-1]] )
        end = relationship[-1]
        #print( f"{relationship} - {start} --> {end} : {count}" )

        col_index = x_axis_list.index(end)
        #print(f"END element {end} is at position {col_index} inside its axis")
        row_index = y_axis_list.index(start)
        #print(f"START element {start} is at position {row_index} inside its axis")
        matrix[row_index, col_index] += count

    #print( matrix )

    if SHOW_PLOT or SAVE_PLOT:
        import matplotlib.pyplot as plt

        # Define some colormap
        from matplotlib.colors import LinearSegmentedColormap
        white_color = (1, 1, 1)
        red_color = (1, 0, 0)
        colors = [white_color, red_color]
        cmap = LinearSegmentedColormap.from_list('white_to_red', list(zip(np.linspace(0, 1, len(colors)), colors)), N=256)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        heatmap = plt.imshow(matrix, cmap=cmap)

        # Add labels and ticks
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('Notes', labelpad=10, loc='right')
        plt.xticks(np.arange(len(x_axis_list)), x_axis_list, rotation=90)
        plt.yticks(np.arange(len(y_axis_list)), [", ".join(list(e)) for e in y_axis_list])
        plt.xlabel('Current note')
        plt.ylabel('Previous Note(s)')

        # Add the value of the counts in the heatmap cells
        for i in range(len(y_axis_list)):
            for j in range(len(x_axis_list)):
                plt.text(j, i, str(int(matrix[i, j])), ha='center', va='center', color='black')

        # Add grid
        # Set minor ticks for grid offset
        plt.gca().set_xticks(np.arange(len(x_axis_list)+1) - 0.5, minor=True)
        plt.gca().set_yticks(np.arange(len(y_axis_list)+1) - 0.5, minor=True)
        #plt.grid(color='lightgray', linestyle='-', linewidth=0.5)  # Aggiungi la griglia
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5, which='minor', alpha=0.5)

        plt.tick_params(axis='both', which='minor', length=0)

        #plt.colorbar(heatmap, label='Relationship counter') #show the legend
        plt.title(f'{MIDIFILENAME[:-4]} - {_DEPTH} order Heatmap')

        if SAVE_PLOT:
            # Save heatmap as PNG
            plt.savefig(f'{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}_{_DEPTH}_order_heatmap.png', bbox_inches='tight')

        if SHOW_PLOT:
            plt.show()

    if SAVE_YAML:
        save_yaml(state_relationships, _DEPTH )

    if _DEPTH == 2 and SAVE_DOT:
        makeDotGraph( state_relationships )

def make1Dgraph( score ):

    # Extract note occurrences from MIDI file
    note_occurrences = {}
    for element in score.recurse(classFilter=('Note', 'Rest') ):
        key = element.nameWithOctave if isinstance(element, note.Note) else 'Rest'

        if key not in note_occurrences:
            note_occurrences[key] = 1
        else:
            note_occurrences[key] += 1

    #print( note_occurrences )
    if SHOW_PLOT or SAVE_PLOT:
        import matplotlib.pyplot as plt
        # Define some colormap
        from matplotlib.colors import LinearSegmentedColormap
        white_color = (1, 1, 1)  # White
        red_color = (1, 0, 0)  # Red
        colors = [white_color, red_color]
        cmap = LinearSegmentedColormap.from_list('white_to_red', list(zip(np.linspace(0, 1, len(colors)), colors)), N=256)


        # Convert note occurrences to matrix format
        notes = sorted( list(note_occurrences.keys()) )
        num_notes = len(notes)
        matrix = np.zeros((1, num_notes))
        for i, note_name in enumerate(notes):
            matrix[0, i] = note_occurrences[note_name]

        # Plot heatmap
        plt.figure(figsize=(10, 3))
        heatmap = plt.imshow(matrix, cmap=cmap, interpolation='nearest')

        # Add labels and ticks
        plt.xticks(np.arange(num_notes), notes)
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('Notes', labelpad=10, loc='right')
        plt.yticks([])
        plt.ylabel('Occurences')
        for i in range(num_notes):
            plt.text(i, 0, str(int(matrix[0, i])), ha='center', va='center', color='black')

        # Add grid
        # Set minor ticks for grid offset
        plt.gca().set_xticks(np.arange(num_notes+1) - 0.5, minor=True)
        #plt.gca().set_yticks(np.arange(len(y_axis_list)+1) - 0.5, minor=True)
        #plt.grid(color='lightgray', linestyle='-', linewidth=0.5)  # Aggiungi la griglia
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5, which='minor', alpha=0.5)

        plt.tick_params(axis='both', which='minor', length=0)

        if SAVE_PLOT:
            # Save heatmap as PNG
            plt.savefig(f'{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}_{1}_order_heatmap.png', bbox_inches='tight')

        if SHOW_PLOT:
            # Show plot
            plt.show()

    if SAVE_YAML:
        save_yaml(note_occurrences, 1 )

# This function will build a dot graph where
# + edges will have a weight according to probability
# + nodes will have the same color if they share the same pitch class (chroma)
# I've just realized that building a dot graph has meaning only for 2nd order relationship.
def makeDotGraph( dictionary ):
    from graphviz import Digraph
    # for every starting state, calculate the total weight of its relationship connections
    total_weight_of_edges = {}
    for relationship, weight in dictionary.items():
        #start_state = "_".join(list( relationship )[:-1])
        start_state, _ = relationship
        if start_state not in total_weight_of_edges:
            total_weight_of_edges[start_state] = weight
        else:
            total_weight_of_edges[start_state] += weight

    #print( total_weight_of_edges )


    # DRAW THE DOT GRAPH *******************************************************
    from matplotlib.colors import LinearSegmentedColormap
    # Defining initial and final colors
    start_color = 'red'
    end_color   = 'yellow'
    # Creating a customised linear colour scale using start and end colours
    cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color])
    # Sampling the colour scale at 12 different equally spaced points
    num_colors = 12
    sampled_colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    # Convert sampled colours to HTML colour specification format
    html_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in sampled_colors]

    # define color for the rests
    REST_COLOR = "gray"

    # Now create a Digraph object
    # ref: https://graphviz.readthedocs.io/en/stable/manual.html
    # ref: https://graphviz.readthedocs.io/en/stable/examples.html
    # take also a look at this:
    # https://networkx.org/documentation/stable/auto_examples/index.html
    dot = Digraph()
    dot.attr(rankdir = 'BT')
    dot.attr(engine='neato')
    dot.attr('node', shape='circle')

    # Add NODE to the graph
    for n in USED_NOTES_LIST:
        if n != 'Rest':
            noteNameNoOctave = n[:-1]
            #print( noteNameNoOctave )
            pithClassIndex = get_pitchClassIndex_from_nomeName( noteNameNoOctave )
            #print( noteNameNoOctave, pithClassIndex )
            color = html_colors[ pithClassIndex ]

            # ref: https://graphviz.org/docs/attrs/pos/
            #pos - Position of node, or spline control points. neato, fdp only.

            vertical_position = (get_midinote_from_noteNameWithOctave(n) - MIN_MIDINOTE) / MIDINOTE_RANGE
            #print( vertical_position )
            # custom vertical position seems not to work
            dot.node(f"{n}", label=f"{n}", pos=f"{0.0},{vertical_position}!", fillcolor=color, style="filled")
        else:
            dot.node(f"{n}", label=f"{n}", fillcolor=REST_COLOR, style="filled")


    # add  EDGES to the graph
    for relationship, weigth in dictionary.items():
        start, end = relationship

        weight_percentage =  round(weigth / total_weight_of_edges[ start ], 2)
        edge_width = weight_percentage * 1
        #dot.edge(f"{previous_name} ({previous_duration})", f"{current_name} ({current_duration})", label=str(count))
        dot.edge(f"{start}", f"{end}", label=str(weight_percentage), fontsize="8", penwidth=str(edge_width))


    # Specify output format (PNG, SVG, etc.)
    output_format = 'png'
    # Save the graph inside the file
    dot.render(f'{OUTPUT_FOLDER}/{MIDIFILENAME[:-4]}_dotgraph', format=output_format, cleanup=True)


# MAIN +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':

    midi_file = converter.parse( f"{MIDIFILES_FOLDER}/{MIDIFILENAME}" )
    #midi_file = midi_file.transpose(12)

    # first of all calculate the used note, this will be useful for all other calculations
    USED_NOTES_LIST = get_list_of_used_notes( midi_file )
    USED_MIDINOTES_LIST = [get_midinote_from_noteNameWithOctave(nnwo) for nnwo in USED_NOTES_LIST if nnwo != 'Rest' ]
    USED_MIDINOTES_LIST = sorted(USED_MIDINOTES_LIST)
    MIN_MIDINOTE = USED_MIDINOTES_LIST[0]
    MAX_MIDINOTE = USED_MIDINOTES_LIST[-1]
    MIDINOTE_RANGE = MAX_MIDINOTE - MIN_MIDINOTE

    for i in range( MAX_ORDER ):
        depth = i+1

        if depth == 1:
            make1Dgraph( midi_file )
        elif depth >= 2:
            make2Dgraph( midi_file, depth )
        else:
            print("ERROR, NUM_ELEMENTS should be equal or greater than 2")
            exit()

    # last thing to do for unknown reason the clean up will brake
    # the other graphs generation if this is done at the beginning.
    if SAVE_SCORE:
        save_musical_score( midi_file )
