Publishing flow:

- commit changes to git
- `bumpversion patch`
- `git push --follow-tags`

PyPi
- python setup.py bdist_wheel
- twine check dist/*
- twine upload dist/*
