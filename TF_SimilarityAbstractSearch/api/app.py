import os
from pathlib import Path
from flask import Flask, request, jsonify, url_for, render_template, redirect
from similarity_abstract_search import utils

app = Flask(__name__)
DATADIR=Path(__file__).resolve().parents[2]/'Data/processed/'
sim_dict = utils.load_json(DATADIR/'sim_vecs.json')
search_dict= utils.load_json(DATADIR/'search.json')
paper_dict=utils.load_json(DATADIR/'paper_data.json')

paperIDtoIDX = {p['id']:idx for idx, p in enumerate(paper_dict)}

# doi_to_idx= get_doi_to_idx()
@app.route("/search", methods=["GET"])
def search():
    q = request.args.get('q','')
    if not q:
        return redirect(url_for('main')) 

    qparts = q.lower().strip().split()
    n = len(paper_dict)
    scores =[]
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in qparts)
        if score ==0: continue
        score += 1.0*(n-i)/n
        scores.append((score, paper_dict[i]))
    scores.sort(reverse=True, key=lambda x: x[0])
    papers = [x[1] for x in scores if x[0]>0]
    if len(papers)>40:
        papers = papers[:40]
    context = default_contex(papers, sort_order='search', search_query=q)
    return render_template('index.html', **context)


@app.route('/sim/<paper_id>')
def sim(paper_id:str=None):
    pidx = paperIDtoIDX.get(paper_id)
    if pidx is None:
        papers=[]
    else:
        sim_ix = sim_dict[pidx]
        papers = [paper_dict[cix] for cix in sim_ix]
    context = default_contex(papers, sort_order='sim')
    return render_template('index.html', **context)

def default_contex(papers, **kwargs):
    gvars = {'num_papers': len(paper_dict)}
    gvars.update(kwargs)
    context = {'papers': papers, 'gvars':gvars}
    return context

@app.route('/')
def main():
    papers = paper_dict[:40]
    context = default_contex(papers, sort_order='latest')
    return render_template('index.html', **context)
    # app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()