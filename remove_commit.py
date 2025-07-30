def callback(commit):
    if commit.original_id == b"66239ac0d0fac701ce47c458bfc876831348d95f":
        commit.skip()
