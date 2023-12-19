brew install python@3.10 \
&& pip install --upgrade pip \
&& pip install virtualenv \
&& virtualenv .venv --python "$(which python3.10)" \
&& source .venv/bin/activate \
&& pip install -r requirements.txt
