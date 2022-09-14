import nox

@nox.session(name="tests")
def tests_coverage(session):
    """ Run pytest and create coverage report.
    """
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.install("codecov")
    session.install("pytest-cov")
    session.run("pytest", "--cov=./", "--cov-report=xml")
    

@nox.session(name="style")
def style_check(session):
    """ Install black and test if the linting is correct.
    """
    session.install("black")
    session.run("black", "--check", "--diff", "tests", "neqr")
