import numpy as np

from voice_agent.tts import load_voice_inventory


def test_voice_inventory_from_npz(tmp_path):
    voices_path = tmp_path / "voices.npz"
    voices = np.array([
        ("kitten/en_female_01", "female", "calm"),
        ("kitten/en_male_01", "male", "bright"),
    ], dtype=object)
    np.savez(voices_path, voices=voices)

    inventory = load_voice_inventory(voices_path)
    assert len(inventory.voices) == 2
    assert inventory.find("kitten/en_male_01").tags == ["male", "bright"]
    assert inventory.default().voice_id == "kitten/en_female_01"
