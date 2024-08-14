import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

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

async function analyzeDependencies() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage("No open text editor found.");
        return;
    }

    const document = editor.document;
    const filePath = document.uri.fsPath;

    const symbolPositions: vscode.Position[] = findImportPositions(document);

    const dependencies: string[] = [];
    for (const position of symbolPositions) {
        const locations = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeDefinitionProvider',
            document.uri,
            position
        );

        if (locations && locations.length > 0) {
            const targetFile = locations[0].uri.fsPath;
            dependencies.push(targetFile);
        }
    }

    visualizeDependencyGraph(filePath, dependencies);
}

function findImportPositions(document: vscode.TextDocument): vscode.Position[] {
    const importPositions: vscode.Position[] = [];
    const text = document.getText();
    const lines = text.split('\n');

    lines.forEach((line, i) => {
        if (line.startsWith('import') || line.startsWith('from')) {
            importPositions.push(new vscode.Position(i, 0));
        }
    });

    return importPositions;
}

function visualizeDependencyGraph(filePath: string, dependencies: string[]) {
    const panel = vscode.window.createWebviewPanel(
        'dependencyGraph',
        'Dependency Graph',
        vscode.ViewColumn.One,
        {}
    );

    const graphData = buildGraphData(filePath, dependencies);
    panel.webview.html = renderGraph(graphData);
}

function buildGraphData(filePath: string, dependencies: string[]) {
    const graph = {
        nodes: [{ id: filePath }],
        edges: dependencies.map(dep => ({ from: filePath, to: dep })),
    };
    return graph;
}

function renderGraph(graphData: any): string {
    // Use a visualization library like D3.js or render simple HTML/SVG.
    return `<html><body><pre>${JSON.stringify(graphData, null, 2)}</pre></body></html>`;
}

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
        vscode.commands.registerCommand('extension.showFraimworkChat', () => {
            const panel = vscode.window.createWebviewPanel(
                'fraimworkChat',
                'Fraimwork Chat',
                vscode.ViewColumn.One,
                {}
            );

            panel.webview.html = getWebviewContent(context);
        })
    );
}


// This method is called when your extension is deactivated
export function deactivate() {}
