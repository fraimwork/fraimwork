import 'dart:convert';
import 'dart:io';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';

void main() {
  String code = stdin.readLineSync(encoding: utf8);

  var parseResult = parseString(content: code);
  var astRoot = parseResult.unit;

  var astJson = astNodeToJson(astRoot);
  print(jsonEncode(astJson));
}

Map<String, dynamic> astNodeToJson(AstNode node) {
  var json = <String, dynamic>{};
  json['type'] = node.runtimeType.toString();
  json['offset'] = node.offset;
  json['length'] = node.length;
  json['children'] = node.childEntities
      .where((child) => child is AstNode)
      .map((child) => astNodeToJson(child as AstNode))
      .toList();
  return json;
}
