import os
from pathlib import Path
from flask import Flask, request, jsonify, url_for, render_template
from similarity_abstract_search import utils

app = Flask(__name__)
DATADIR=Path(__file__).resolve().parents[3]/'Data/processed/'
sim_dict = utils.load_json(DATADIR/'sim_vecs.json')
search_dict= utils.load_json(DATADIR/'search.json')
raw_paper_data=utils.load_json(DATADIR/'raw_paper_data.json')

# doi_to_idx= get_doi_to_idx()
@app.route("/search", methods=["GET"])
def search():
    pass

@app.route('/sim/<doi_prefix>/doi_suffix')
def sim(doi_prefix=None, doi_suffix=None):
    pass
    # return render_template('index.html', **context)

def main():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()