import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { Graph } from 'graphlib';

const workspaceFolders = vscode.workspace.workspaceFolders;

if (workspaceFolders) {
    workspaceFolders.forEach((folder: { uri: { fsPath: any; }; }) => {
        console.log('Folder: ', folder.uri.fsPath);
    });
}

function getWebviewContent(context: vscode.ExtensionContext): string {
    const htmlFilePath = path.join(context.extensionPath, 'media', 'webview.html');
    const htmlContent = fs.readFileSync(htmlFilePath, 'utf8');
    return htmlContent;
}

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('fraimwork.examineDependencies', async () => {
        vscode.window.showInformationMessage('Analyzing dependencies...');

        const files = await getFiles();
        console.log('Files: ', files);
        const map: { [key: string]: string[] } = {};

        const promises = files.map(file => findDependencies(file));
        const results = await Promise.all(promises);
        results.forEach((dependencies, i) => {
            const relativePath = path.relative(workspaceFolders![0].uri.fsPath, files[i].fsPath);
            map[relativePath] = Array.from(dependencies);
        });

        const graph = new Graph();
        // Add notes from map
        for (const filePath in map) {
            const fileName = path.basename(filePath);
            graph.setNode(filePath, { label: fileName });
        }
        // Add edges
        for (const filePath in map) {
            for (const dep of map[filePath]) {
                graph.setEdge(filePath, dep);
            }
        }

        visualizeDependencyGraph(graph);
    });

    context.subscriptions.push(disposable);
}

async function findDependencies(file: vscode.Uri): Promise<Set<string>> {
    const dependencies = new Set<string>();
    
    const document = await vscode.workspace.openTextDocument(file);
    const text = document.getText();
    const lines = text.split('\n');
    if (lines) {
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (!line.trim().startsWith('import') && !line.trim().startsWith('from')) {
                continue;
            }
            const tokens = line.split(' ');

            for (const token of tokens) {
                console.log(token);
                const location = new vscode.Position(i, line.indexOf(token) + token.length/2);

                // Get definition using the language server
                const locations = await vscode.commands.executeCommand<vscode.LocationLink[]>(
                    'vscode.executeDefinitionProvider',
                    file,
                    location
                );

                if (locations && locations.length > 0) {
                    try {
                        const dependencyFile = locations[0].targetUri.fsPath;
                        // Ensure the dependency file is not the current file and that it is in the same workspace
                        if (dependencyFile !== file.fsPath && workspaceFolders && workspaceFolders.some((folder) => dependencyFile.startsWith(folder.uri.fsPath))) {
                            const relativePath = path.relative(workspaceFolders[0].uri.fsPath, dependencyFile);
                            // Get symbols within dependency
                            const dependencyUri = vscode.Uri.file(dependencyFile);
                            if (dependencyUri) {
                                const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                                    'vscode.executeDocumentSymbolProvider',
                                    dependencyUri
                                );
                                if (symbols) {
                                    for (const symbol of symbols) {
                                        if (symbol.kind == 5 || symbol.kind == 12 || symbol.kind == 1) {
                                            const symbolKind = SymbolKindMap[symbol.kind] || "Unknown";
                                            dependencies.add(`${relativePath}#${symbol.name} (${symbolKind})`);
                                        } else {
                                            dependencies.add(dependencyFile);
                                        }
                                    }
                                } 
                            }
                            
                        }
                    } catch (error) {
                        console.log("Error retrieving file " +  error);
                    }
                }
            }
        }
    }

    return dependencies;
}


async function getFiles(): Promise<vscode.Uri[]> {
    const pattern = '**/*.{js,jsx,ts,tsx,py,dart}';
    const ignoredPattern = '**/node_modules/**'; // Adjust this pattern based on the file types you're interested in
    const files = await vscode.workspace.findFiles(pattern, ignoredPattern);
    return files;
}

async function getSymbols(uri: vscode.Uri): Promise<vscode.DocumentSymbol[]> {
    const document = await vscode.workspace.openTextDocument(uri);
    const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>('vscode.executeDocumentSymbolProvider', document.uri);

    return symbols || [];
}

function visualizeDependencyGraph(dependencyGraph: Graph) {
    const panel = vscode.window.createWebviewPanel(
        'dependencyGraph',
        'Dependency Graph',
        vscode.ViewColumn.One,
        {}
    );

    const [nodes, edges] = [dependencyGraph.nodes(), dependencyGraph.edges()];
    const graphData = { nodes, edges };
    panel.webview.html = renderGraph(graphData);
}

function renderGraph(graphData: any): string {
    // Use a visualization library like D3.js or render simple HTML/SVG.
    return `<html><body><pre>${JSON.stringify(graphData, null, 2)}</pre></body></html>`;
}

// This method is called when your extension is deactivated
export function deactivate() {}

// Based on documentation provided by vscode: https://vscode-api.js.org/enums/vscode.SymbolKind.html
const SymbolKindMap: { [key: number]: string } = {
    1: 'File',
    2: 'Module',
    3: 'Namespace',
    4: 'Package',
    5: 'Class',
    6: 'Method',
    7: 'Property',
    8: 'Field',
    9: 'Constructor',
    10: 'Enum',
    11: 'Interface',
    12: 'Function',
};