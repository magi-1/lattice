import nox

locations = "lattice", "noxfile.py"


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
