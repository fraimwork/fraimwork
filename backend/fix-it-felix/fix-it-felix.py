from flask import Flask, request, jsonify
from fixing import fix_broken_code

app = Flask(__name__)

@app.route('/fix', methods=['POST'])
def fix_code():
    try:
        # Receive the code dependency graph and broken nodes
        data = request.get_json()
        dependency_graph = data.get('dependency_graph')
        broken_nodes = data.get('broken_nodes')
        if not dependency_graph or not broken_nodes:
            return jsonify({"error": "Missing 'dependency_graph' or 'broken_nodes' in request."}), 400

        # Call the function to fix the broken code (to be implemented)
        fixed_code = fix_broken_code(dependency_graph, broken_nodes)

        
        return jsonify({"fixed_code": fixed_code}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)