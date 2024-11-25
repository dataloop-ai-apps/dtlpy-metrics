Publishing flow:

- build docker image
- update manifest with new image URL
- commit changes to git
- `bumpversion patch`
- `git push --follow-tags`
- submit PR
- merge PR


PyPi
- python setup.py bdist_wheel
- twine check dist/*
- twine upload dist/*


DPK
- run publish