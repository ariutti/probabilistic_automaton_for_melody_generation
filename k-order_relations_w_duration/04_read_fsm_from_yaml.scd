// load an existing FSM (in YAML format) inside a SC dictionary
(
// READ ALL THE EXHISTING YAML FILE INSIDE THE FOLDER GIVEN THE NAME OF THE SONG
~yaml_folder_pathString = PathName(thisProcess.nowExecutingPath).pathOnly +/+ "renders";
~file_name = "old_mc_donald";
{
	d = Dictionary.new();
	PathName(~yaml_folder_pathString).entries.do({
		| entry |
		if( (entry.isFile).and(entry.extension == "yaml").and(entry.fileNameWithoutExtension.contains( ~file_name) ), {
			var num_relationship = entry.fullPath.split($_).rotate(2).last.asSymbol;
			//entry.fullPath.postln;
			var f = File( entry.fullPath, "r");
			d.add( num_relationship -> f.readAllString.parseYAML); // a dictionary

		});
	});
}.value;
)

// number of state machine we collected inside the dictionary
// (corresponds to the actual number of yaml file inside the folder)
d.size;

d.postln
d[\2]

(
// a function to 'pretty-print' the fsm of given order inside the Post window
~func_print_fsm = {
	|fsm|
	var num_start_nodes = fsm.keys.size;
	"\nPrinting out the fsm:".postln;
	fsm.keysValuesDo({
		|k, v|
		//[k, v].postln;
		// k represent a possible start state;
		// v represent the possibles end states for the current start state;
		"start state '%'\n".postf(k);
		v.pairsDo({
			|endPoint, weight|
			"\tend state '%' (weigth: %)\n".postf(endPoint, weight);
		})
	});
}
);

~func_print_fsm.value( d[\1] ); // print 1st order fsm
~func_print_fsm.value( d[\2] ); // print 2nd order fsm
~func_print_fsm.value( d[\3] ); // print 3rd order fsm
~func_print_fsm.value( d[\4] ); // print 4th order fsm
~func_print_fsm.value( d[\5] ); // print 5th order fsm


(
// given a fsm, return one of the possibles end node for
// a given starting node
~func_get_end_node = {
	|currentState, fsm|
	//[currentState, fsm].postln;
	// check if current state is a possible start node of the fsm graph
	if( fsm.includesKey( currentState ), {
		var possible_outcomes = fsm[ currentState ];
		var states  = [];
		var weights = [];
		possible_outcomes.pairsDo({
			|s,w|
			states = states.add(s);
			weights = weights.add(w.asInteger);
		});
		//states.postln;
		//weights.postln;
		states.wchoose( weights.normalizeSum );
	}, {
		"WARNING: given start state '%' is not present in given fsm.\n".postf( currentState );
		//return the first state in the fsm
		fsm;
	});
}
);
~func_get_end_node.("F#4_quarter", d[\2]);
~func_get_end_node.("F#5_quarter_F#5_half", d[\3]);



(
~func_get_dur_from_music21_durs = {
	|music21_dur|
	switch( music21_dur,
		"16th",   { 0.25; },
		"eighth", { 0.5;  },
		"quarter",{ 1;  },
		"half",   { 2;  },
		"whole",  { 4;  },
		{
			"dur not defined".postln;
			1;
		}
	);
}
)
//~func_get_dur_from_music21_durs.("quarter");

(
~func_get_midinote_from_noteNameWithOctave_duration = {
	|noteNameAllwDur|
	var noteNameAll = noteNameAllwDur.split($_)[0];
	if( noteNameAll != "Rest", {
		var noteName = noteNameAll[0].asString;
		var octNumber = noteNameAll.rotate(1)[0].asString.asInteger;
		var midiNumber = (octNumber +1) * 12;
		switch( noteName.toLower,
			"c", { midiNumber = midiNumber+0; },
			"d", { midiNumber = midiNumber+2; },
			"e", { midiNumber = midiNumber+4; },
			"f", { midiNumber = midiNumber+5; },
			"g", { midiNumber = midiNumber+7; },
			"a", { midiNumber = midiNumber+9; },
			"b", { midiNumber = midiNumber+11; },
		);

		// music21 seems to use '#' for sharps and '-' for flats
		if( noteNameAll.contains($#), {midiNumber = midiNumber + 1;});
		if( noteNameAll.contains($-), {midiNumber = midiNumber - 1;});

		midiNumber;
	}, {
		"rest".asSymbol;
	}
);
};
)
//~func_get_midinote_from_noteNameWithOctave_duration.("E#1_quarter");

(
~func_insert_new_element_in_list_and_get_most_recent_depth_notes = {
	|element, list, depth|
	// remove oldest element
	list.removeAt(0);
	// insert the current element
	// as the last element (rightmost)
	list.insert( depth-1, element );
	list.postln;
};
)

(
~func_compose_key_from_list = {
	|list|
	// remove 'nil' elements
	// 'nil' items will be at firsts indexes during the first iterations.
	var filteredList = list.reject({|e,i| e.isNil});
	// than compose a unique string made of the element
	// of the filtered list joined with an 'underscore'
	filteredList.join($_);
}
);


// TEST THE FSM
s.boot;
t = TempoClock.new(90/60);


// Try execute this block of code while the Pbindef is running and maybe change the
// '~max_order' value o change the "randomsess" of the execution:
// + the greater value will "force" execution to be more and more similar to the original one;
// + sthe maller values instead will make the execution to be more free, but still maintain
//   a sort of similarity with the originale material!
(
~events_counter = 0;
~start_node   = "undefined";
// the following value represent what is the max depth of the "start state"
// for example, if the more complicated state machine we are using has its
// start states in the form of ,say, 'F#4_rest_B-5'
// it means that the max depth must be set to '3'
~max_order = 1;
~last_played_notes_list = Array.newClear(~max_order);
)


(
Pbindef(\test_fms,
	\instrument, \default,
	\amp, 0.1,
	\end_node, Pfunc({
		// select the not to be played:
		// + if this is the first note to be played, simply select the current state
		//   from the probabilities array (saved inside the first fsm yaml file);
		// + if this is the second, third and so on, select the corresponding state machine
		//   to get the next state from;
		var end_node, state_machine_index;
		if( ~events_counter == 0, {
			"first iteration".postln;
		});

		state_machine_index = (~events_counter+1).clip(1, ~max_order+1);
		end_node = ~func_get_end_node.(~start_node, d[ state_machine_index.asSymbol ]);
		/*
		var end_node = switch( ~events_counter,
			0, {
				//var noteNameWithOctaveString =
				//~func_select_first_state.(~current_state);
			},
			1, {
				~func_get_end_node.(~start_node, d[2]);
			},
			{
				//"deafult state".postln;
				~func_get_end_node.(~start_node, d[3]);
		});
		*/
		"current end node is: %\n".postf( end_node );
		end_node;
	}),


	\midinote, Pfunc({
		|e|
		var end_node = e.use{
			~end_node.();
		};
		~func_get_midinote_from_noteNameWithOctave_duration.( end_node );
	}),



	// variable 'dur' to be later implemented in nexct updates
	\dur, Pfunc({
		|e|
		var a = e.use{
			~end_node.().split($_);
		};
		~func_get_dur_from_music21_durs.(a[1]);
	}),


	//\dur, 1,

	\finish, Pfunc({
		|e|
		var tmp_list;
		var end_node = e.use{
			~end_node.();
		};
		// the current end node will become the next start node for the next state machine
		"current transition was between % -> %\n".postf( ~start_node, end_node );
		tmp_list = ~func_insert_new_element_in_list_and_get_most_recent_depth_notes.value(end_node, ~last_played_notes_list, ~max_order);
		// compute the next 'start_node'
		~start_node = ~func_compose_key_from_list.value( tmp_list );
		"next Start_state is %\n".postf( ~start_node );
		// update number of events played
		~events_counter = ~events_counter + 1;
	})
).quant_([1,0.1]).stop.play(t);
)
Pbindef(\test_fms).stop.clear;


// TODO:
// * inserisci nella macchina a stati quale sia lo stato inziale;
// * definisci anche quali siano i possibili stati finali;
