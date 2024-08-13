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
