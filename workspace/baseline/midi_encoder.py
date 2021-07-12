import os
import argparse
import numpy   as np
import math    as ma
import music21 as m21

THREE_DOTTED_BREVE = 15
THREE_DOTTED_32ND  = 0.21875

MIN_VELOCITY = 0
MAX_VELOCITY = 128

MIN_TEMPO = 24
MAX_TEMPO = 160

MAX_PITCH = 128

def load(datapath, sample_freq=4, piano_range=(33, 93), transpose_range=10, stretching_range=10):
    text = ""
    vocab = set()

    if os.path.isfile(datapath):
        # Path is an individual midi file
        file_extension = os.path.splitext(datapath)[1]

        if file_extension == ".midi" or file_extension == ".mid":
            text = parse_midi(datapath, sample_freq, piano_range, transpose_range, stretching_range)
            vocab = set(text.split(" "))
    else:
        # Read every file in the given directory
        for file in os.listdir(datapath):
            file_path = os.path.join(datapath, file)
            file_extension = os.path.splitext(file_path)[1]

            # Check if it is not a directory and if it has either .midi or .mid extentions
            if os.path.isfile(file_path) and (file_extension == ".midi" or file_extension == ".mid"):
                encoded_midi = parse_midi(file_path, sample_freq, piano_range, transpose_range, stretching_range)

                if len(encoded_midi) > 0:
                    words = set(encoded_midi.split(" "))
                    vocab = vocab | words

                    text += encoded_midi + " "

        # Remove last space
        text = text[:-1]

    return text, vocab

def parse_midi(file_path, sample_freq, piano_range, transpose_range, stretching_range):

    # Split datapath into dir and filename
    midi_dir = os.path.dirname(file_path)
    midi_name = os.path.basename(file_path).split(".")[0]

    # If txt version of the midi already exists, load data from it
    midi_txt_name = os.path.join(midi_dir, midi_name + ".txt")

    if(os.path.isfile(midi_txt_name)):
        midi_fp = open(midi_txt_name, "r")
        encoded_midi = midi_fp.read()
    else:
        # Create a music21 stream and open the midi file
        midi = m21.midi.MidiFile()
        midi.open(file_path)
        midi.read()
        midi.close()

        # Translate midi to stream of notes and chords
        encoded_midi = midi2encoding(midi, sample_freq, piano_range, transpose_range, stretching_range)

        if len(encoded_midi) > 0:
            midi_fp = open(midi_txt_name, "w+")
            midi_fp.write(encoded_midi)
            midi_fp.flush()

    midi_fp.close()
    return encoded_midi

def midi2encoding(midi, sample_freq, piano_range, transpose_range, stretching_range):
    try:
        midi_stream = m21.midi.translate.midiFileToStream(midi)
    except:
        return []

    # Get piano roll from midi stream
    piano_roll = midi2piano_roll(midi_stream, sample_freq, piano_range, transpose_range, stretching_range)

    # Get encoded midi from piano roll
    encoded_midi = piano_roll2encoding(piano_roll)

    return " ".join(encoded_midi)

def piano_roll2encoding(piano_roll):
    # Transform piano roll into a list of notes in string format
    final_encoding = {}

    perform_i = 0
    for version in piano_roll:
        lastTempo    = -1
        lastVelocity = -1
        lastDuration = -1.0

        version_encoding = []

        for i in range(len(version)):
            # Time events are stored at the last row
            tempo = version[i,-1][0]
            if tempo != 0 and tempo != lastTempo:
                version_encoding.append("t_" + str(int(tempo)))
                lastTempo = tempo

            # Process current time step of the piano_roll
            for j in range(len(version[i]) - 1):
                duration = version[i,j][0]
                velocity = int(version[i,j][1])

                if velocity != 0 and velocity != lastVelocity:
                    version_encoding.append("v_" + str(velocity))
                    lastVelocity = velocity

                if duration != 0 and duration != lastDuration:
                    duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
                    version_encoding.append("d_" + duration_tuple.type + "_" + str(duration_tuple.dots))
                    lastDuration = duration

                if duration != 0 and velocity != 0:
                    version_encoding.append("n_" + str(j))

            # End of time step
            if len(version_encoding) > 0 and version_encoding[-1][0] == "w":
                # Increase wait by one
                version_encoding[-1] = "w_" + str(int(version_encoding[-1].split("_")[1]) + 1)
            else:
                version_encoding.append("w_1")

        # End of piece
        version_encoding.append("\n")

        # Check if this version of the MIDI is already added
        version_encoding_str = " ".join(version_encoding)
        if version_encoding_str not in final_encoding:
            final_encoding[version_encoding_str] = perform_i

        perform_i += 1

    return final_encoding.keys()

def write(encoded_midi, path):
    # Base class checks if output path exists
    midi = encoding2midi(encoded_midi)
    midi.open(path, "wb")
    midi.write()
    midi.close()

def encoding2midi(note_encoding, ts_duration=0.25):
    notes = []

    velocity = 100
    duration = "16th"
    dots = 0

    ts = 0
    for note in note_encoding.split(" "):
        if len(note) == 0:
            continue

        elif note[0] == "w":
            wait_count = int(note.split("_")[1])
            ts += wait_count

        elif note[0] == "n":
            pitch = int(note.split("_")[1])
            note = m21.note.Note(pitch)
            note.duration = m21.duration.Duration(type=duration, dots=dots)
            note.offset = ts * ts_duration
            note.volume.velocity = velocity
            notes.append(note)

        elif note[0] == "d":
            duration = note.split("_")[1]
            dots = int(note.split("_")[2])

        elif note[0] == "v":
            velocity = int(note.split("_")[1])

        elif note[0] == "t":
            tempo = int(note.split("_")[1])

            if tempo > 0:
                mark = m21.tempo.MetronomeMark(number=tempo)
                mark.offset = ts * ts_duration
                notes.append(mark)

    piano = m21.instrument.fromString("Piano")
    notes.insert(0, piano)

    piano_stream = m21.stream.Stream(notes)
    main_stream  = m21.stream.Stream([piano_stream])

    return m21.midi.translate.streamToMidiFile(main_stream)

def midi_parse_notes(midi_stream, sample_freq):
    note_filter = m21.stream.filters.ClassFilter('Note')

    note_events = []
    for note in midi_stream.recurse().addFilter(note_filter):
        pitch    = note.pitch.midi
        duration = note.duration.quarterLength
        velocity = note.volume.velocity
        offset   = ma.floor(note.offset * sample_freq)

        note_events.append((pitch, duration, velocity, offset))

    return note_events

def midi_parse_chords(midi_stream, sample_freq):
    chord_filter = m21.stream.filters.ClassFilter('Chord')

    note_events = []
    for chord in midi_stream.recurse().addFilter(chord_filter):
        pitches_in_chord = chord.pitches
        for pitch in pitches_in_chord:
            pitch    = pitch.midi
            duration = chord.duration.quarterLength
            velocity = chord.volume.velocity
            offset   = ma.floor(chord.offset * sample_freq)

            note_events.append((pitch, duration, velocity, offset))

    return note_events

def midi_parse_metronome(midi_stream, sample_freq):
    metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

    time_events = []
    for metro in midi_stream.recurse().addFilter(metronome_filter):
        time = int(metro.number)
        offset = ma.floor(metro.offset * sample_freq)
        time_events.append((time, offset))

    return time_events

def midi2notes(midi_stream, sample_freq, transpose_range):
    notes = []
    notes += midi_parse_notes(midi_stream, sample_freq)
    notes += midi_parse_chords(midi_stream, sample_freq)

    # Transpose the notes to all the keys in transpose_range
    return transpose_notes(notes, transpose_range)

def midi2piano_roll(midi_stream, sample_freq, piano_range, transpose_range, stretching_range):
    # Calculate the amount of time steps in the piano roll
    time_steps = ma.floor(midi_stream.duration.quarterLength * sample_freq) + 1

    # Parse the midi file into a list of notes (pitch, duration, velocity, offset)
    transpositions = midi2notes(midi_stream, sample_freq, transpose_range)

    time_events = midi_parse_metronome(midi_stream, sample_freq)
    time_streches = strech_time(time_events, stretching_range)

    return notes2piano_roll(transpositions, time_streches, time_steps, piano_range)

def notes2piano_roll(transpositions, time_streches, time_steps, piano_range):
    performances = []

    min_pitch, max_pitch = piano_range
    for t_ix in range(len(transpositions)):
        for s_ix in range(len(time_streches)):
            # Create piano roll with calcualted size.
            # Add one dimension to very entry to store velocity and duration.
            piano_roll = np.zeros((time_steps, MAX_PITCH + 1, 2))

            for note in transpositions[t_ix]:
                pitch, duration, velocity, offset = note
                if duration == 0.0:
                    continue

                # Force notes to be inside the specified piano_range
                pitch = clamp_pitch(pitch, max_pitch, min_pitch)

                piano_roll[offset, pitch][0] = clamp_duration(duration)
                piano_roll[offset, pitch][1] = discretize_value(velocity, bins=32, range=(MIN_VELOCITY, MAX_VELOCITY))

            for time_event in time_streches[s_ix]:
                time, offset = time_event
                piano_roll[offset, -1][0] = discretize_value(time, bins=100, range=(MIN_TEMPO, MAX_TEMPO))

            performances.append(piano_roll)

    return performances

def transpose_notes(notes, transpose_range):
    transpositions = []

    # Modulate the piano_roll for other keys
    first_key = -ma.floor(transpose_range/2)
    last_key  =  ma.ceil(transpose_range/2)

    for key in range(first_key, last_key):
        notes_in_key = []
        for n in notes:
            pitch, duration, velocity, offset = n
            t_pitch = pitch + key
            notes_in_key.append((t_pitch, duration, velocity, offset))
        transpositions.append(notes_in_key)

    return transpositions

def strech_time(time_events, stretching_range):
    streches = []

    # Modulate the piano_roll for other keys
    slower_time = -ma.floor(stretching_range/2)
    faster_time =  ma.ceil(stretching_range/2)

    # Modulate the piano_roll for other keys
    for t_strech in range(slower_time, faster_time):
        time_events_in_strech = []
        for t_ev in time_events:
            time, offset = t_ev
            s_time = time + 0.05 * t_strech * MAX_TEMPO
            time_events_in_strech.append((s_time, offset))
        streches.append(time_events_in_strech)

    return streches

def discretize_value(val, bins, range):
    min_val, max_val = range

    val = int(max(min_val, val))
    val = int(min(val, max_val))

    bin_size = (max_val/bins)
    return ma.floor(val/bin_size) * bin_size

def clamp_pitch(pitch, max, min):
    while pitch < min:
        pitch += 12
    while pitch >= max:
        pitch -= 12
    return pitch

def clamp_duration(duration, max=THREE_DOTTED_BREVE, min=THREE_DOTTED_32ND):
    # Max duration is 3-dotted breve
    if duration > max:
        duration = max

    # min duration is 3-dotted breve
    if duration < min:
        duration = min

    duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
    if duration_tuple.type == "inexpressible":
        duration_clossest_type = m21.duration.quarterLengthToClosestType(duration)[0]
        duration = m21.duration.typeToDuration[duration_clossest_type]

    return duration

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_encoder.py')
    parser.add_argument('--path', type=str, required=True, help="Path to midi data.")
    parser.add_argument('--transp', type=int, default=1, help="Transpose range.")
    parser.add_argument('--strech', type=int, default=1, help="Time stretching range.")
    opt = parser.parse_args()

    # Load data and encoded it
    text, vocab = load(opt.path, transpose_range=opt.transp, stretching_range=opt.strech)
    print(text)

    # Write all data to midi file
    write(text, "encoded.mid")
