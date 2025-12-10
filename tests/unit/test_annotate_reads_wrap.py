import inspect

import wrappers.annotate_reads_wrap as ann


def test_include_flags_defaults_off():
    sig = inspect.signature(ann.annotate_reads_wrap)
    assert sig.parameters['include_edit_distances'].default is False
    assert sig.parameters['include_sequences_in_valid_output'].default is False
