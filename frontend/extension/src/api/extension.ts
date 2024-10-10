import * as vscode from 'vscode';
import { initializeApp } from "firebase/app";
import { getAuth, signInWithPopup, GithubAuthProvider } from "firebase/auth";
import { app, auth, provider } from './firebase';
import path from 'path';

export function activate(context: vscode.ExtensionContext) {
    let panel: vscode.WebviewPanel | undefined = undefined;

    context.subscriptions.push(
        vscode.commands.registerCommand('chatExtension.openChat', () => {
            if (panel) {
                panel.reveal(vscode.ViewColumn.One);
            } else {
                panel = vscode.window.createWebviewPanel(
                    'chat',
                    'Chat Interface',
                    vscode.ViewColumn.One,
                    {
                        enableScripts: true
                    }
                );

                panel.webview.html = getWebviewContent(context);

                panel.webview.onDidReceiveMessage(
                    async message => {
                        switch (message.type) {
                            case 'chat':
                                const response = await sendChatMessage(message.text);
                                panel?.webview.postMessage({ text: response });
                                break;
                        }
                    },
                    undefined,
                    context.subscriptions
                );

                panel.onDidDispose(
                    () => {
                        panel = undefined;
                    },
                    null,
                    context.subscriptions
                );
            }
        })
    );

    async function sendChatMessage(text: string): Promise<string> {
        // Replace with the backend API call
        const response = await fetch('https://backend-api.com/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${await getFirebaseToken()}`
            },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();

        if (typeof data === 'object' && data !== null && 'reply' in data) {
            return (data as { reply: string }).reply;
        } else {
            throw new Error('Unexpected response format');
        }
    }

    async function getFirebaseToken(): Promise<string> {
        const user = auth.currentUser;
        if (user) {
            return user.getIdToken();
        } else {
            const result = await signInWithPopup(auth, provider);
            return result.user.getIdToken();
        }
    }
}

function getWebviewContent(context: vscode.ExtensionContext) {
    const chatHtmlPath = vscode.Uri.file(
        path.join(context.extensionPath, 'media', 'chat.html')
    );
    const chatHtmlUri = chatHtmlPath.with({ scheme: 'vscode-resource' });
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat Interface</title>
            <style>
                body { font-family: Arial, sans-serif; }
                #chat { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
                #input { width: 100%; padding: 10px; }
            </style>
        </head>
        <body>
            <div id="chat"></div>
            <input type="text" id="input" placeholder="Type a message..." />
            <script>
                const vscode = acquireVsCodeApi();

                document.getElementById('input').addEventListener('keydown', function(event) {
                    if (event.key === 'Enter') {
                        const message = event.target.value;
                        vscode.postMessage({ type: 'chat', text: message });
                        event.target.value = '';
                    }
                });

                window.addEventListener('message', event => {
                    const message = event.data;
                    const chat = document.getElementById('chat');
                    const messageElement = document.createElement('div');
                    messageElement.textContent = message.text;
                    chat.appendChild(messageElement);
                    chat.scrollTop = chat.scrollHeight;
                });
            </script>
        </body>
        </html>
    `;
}
