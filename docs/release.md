Publishing flow:

- commit changes to git
- `bumpversion patch`
- `git push --follow-tags`

PyPi
- python setup.py bdist_wheel
- twine check dist/*
- twine upload dist/*

DPK
- python create_app.py --publish --project 4eff4f18-5285-42bb-b43d-f74cea633916