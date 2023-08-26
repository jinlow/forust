# Building with polars is sooo slow.
# It's only there for the example, so let's remove it
# in the regular build process.
# Requires toml package
import toml

ct = toml.load("../cargo.toml")

del ct["dev-dependencies"]
del ct["bench"]

with open("../cargo.toml", "w") as file:
    toml.dump(ct, file)
