import nox

locations = "lattice", "tests", "noxfile.py"


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
