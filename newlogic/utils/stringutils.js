const _ = require('lodash');
const re = require('re');
const { lruCache } = require('lodash');
const { default: difflib } = require('re');

const arrFromSepString = (string, sep = ',') => string.split(sep).map(x => x.trim());

const arrFromNumberedList = (string) => arrFromSepString(string, "\n").map(x => x.split(' ')[1]);

const extractMarkdownBlocks = (text) => {
    const pattern = /```(.*?)\n(.*?)```/gs;
    const blocks = Array.from(text.matchAll(pattern)).map(match => match[2].trim());
    return blocks;
};

const markdownToDict = (markdown) => {
    const headerRegex = /^#+\s*(.*)/gm;
    const result = {};
    let match;
    const headers = [];

    while ((match = headerRegex.exec(markdown)) !== null) {
        headers.push([match[0].length, match[1], match.index]);
    }

    headers.sort((a, b) => a[2] - b[2]);

    for (let i = 0; i < headers.length; i++) {
        const [headerLevel, headerText, headerStart] = headers[i];
        const nextHeaderStart = i + 1 < headers.length ? headers[i + 1][2] : markdown.length;
        const content = markdown.slice(headerStart + headerText.length + headerLevel, nextHeaderStart).trim();
        result[headerText.toLowerCase()] = content;
    }

    return result;
};

const wordwiseTokenize = (text) => text.match(/[a-zA-Z]+|\s+|[^a-zA-Z\s]+/g) || [];

const linewiseTokenize = (text) => text.split(/\r?\n/);

const _editDistance = _.memoize((str1, str2) => {
    const m = str1.length, n = str2.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) {
        for (let j = 0; j <= n; j++) {
            if (i === 0) dp[i][j] = j;
            else if (j === 0) dp[i][j] = i;
            else dp[i][j] = str1[i - 1] === str2[j - 1] ? dp[i - 1][j - 1] : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
        }
    }
    return dp[m][n];
});

const editDistanceCache = new Map();
const editDistance = (list1, list2) => {
    const key = [list1, list2].toString();
    if (editDistanceCache.has(key)) return editDistanceCache.get(key);
    const m = list1.length, n = list2.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) {
        for (let j = 0; j <= n; j++) {
            if (i === 0) dp[i][j] = _.sum(_.range(j).map(k => _editDistance('', list2[k])));
            else if (j === 0) dp[i][j] = _.sum(_.range(i).map(k => _editDistance(list1[k], '')));
            else dp[i][j] = Math.min(dp[i - 1][j] + _editDistance(list1[i - 1], ''), dp[i][j - 1] + _editDistance('', list2[j - 1]), dp[i - 1][j - 1] + _editDistance(list1[i - 1], list2[j - 1]));
        }
    }
    editDistanceCache.set(key, dp[m][n]);
    return dp[m][n];
};

// Function to calculate the score based on edit distance
const score = (a, b) => {
    const maxLen = Math.max(a.join('').length, b.join('').length);
    if (maxLen === 0) return 1;
    const diff = editDistance(a, b);
    return 1 - diff / maxLen;
};

const weightedScore = (a, b) => score(a, b) * _.mean([a.length, b.length]);


let dbKmersCache = {};

function ktupleMatching(query, database, k, partProcessing = (x) => x) {
    let databaseKmers;

    const cacheKey = [database.join(''), k].toString();
    if (cacheKey in dbKmersCache) {
        databaseKmers = dbKmersCache[cacheKey];
    } else {
        databaseKmers = {};
        for (let i = 0; i <= database.length - k; i++) {
            let kmer = database.slice(i, i + k).map(partProcessing).join('');
            if (kmer === '') continue;
            if (!databaseKmers[kmer]) {
                databaseKmers[kmer] = [];
            }
            databaseKmers[kmer].push(i);
        }
        dbKmersCache[cacheKey] = databaseKmers;
    }

    let matches = [];
    for (let i = 0; i <= query.length - k; i++) {
        let kmer = query.slice(i, i + k).map(partProcessing).join('');
        if (kmer === '') continue;
        if (kmer in databaseKmers) {
            for (let dbPos of databaseKmers[kmer]) {
                matches.push([i, dbPos]);
            }
        }
    }
    return matches;
}

function findTopDiagonal(matches, query, database, k) {
    let diagonalScores = {};
    for (let [qPos, dbPos] of matches) {
        let diag = qPos - dbPos;
        if (!(diag in diagonalScores)) {
            diagonalScores[diag] = 0;
        }
        diagonalScores[diag] += query.slice(qPos, qPos + k)
            .map((q, i) => weightedScore(q, database[dbPos + i]))
            .reduce((a, b) => a + b, 0) / k;
    }
    let topDiagonal = Object.entries(diagonalScores).reduce(
        (maxEntry, entry) => entry[1] > maxEntry[1] ? entry : maxEntry, [null, 0]
    );
    return topDiagonal[0];
}

function smithWatermanDiagonal({
    query, database, matchScore = 3, mismatchPenalty = 3, gapPenalty = 2, 
    processing = null, tolerance = 0.75, diag = 0, bandWidth = 5, 
    dbTokenDepthRoc = null, queryTokenDepthRoc = null
}) {
    const m = query.length, n = database.length;
    let scoringMatrix = {}, tracebackMatrix = {};
    let startRow = diag >= 0 ? 1 : diag;
    let startCol = diag >= 0 ? -diag : 1;

    let maxScore = 0;
    let maxPos = [0, 0];
    let covered = [];

    while (startRow <= m && startCol <= n) {
        for (let offset = -bandWidth; offset <= bandWidth; offset++) {
            let row = startRow, col = startCol + offset;
            if (!(0 < row && row <= m && 0 < col && col <= n)) continue;

            let q = query[row - 1], d = database[col - 1];
            covered.push([row - 1, col - 1]);

            if (processing) {
                q = processing(q);
                d = processing(d);
            }
            let multiplier = 1 + (q.length + d.length) / 2;
            let match = (scoringMatrix[[row - 1, col - 1]] || 0) + 
                        (score(q, d) >= tolerance ? matchScore : -mismatchPenalty) * multiplier;
            let insert = (scoringMatrix[[row, col - 1]] || 0) - gapPenalty * multiplier;

            scoringMatrix[[row, col]] = Math.max(match, insert);

            if (scoringMatrix[[row, col]] === match) {
                tracebackMatrix[[row, col]] = 1;
            } else if (scoringMatrix[[row, col]] === insert) {
                tracebackMatrix[[row, col]] = 3;
            }

            if (scoringMatrix[[row, col]] >= maxScore) {
                maxScore = scoringMatrix[[row, col]];
                maxPos = [row, col];
            }
        }
        startRow++;
        startCol++;
    }

    let align1 = [], align2 = [];
    let [i, j] = maxPos;
    while ((scoringMatrix[[i, j]] || 0) > 0) {
        let q = query[i - 1], d = database[j - 1];
        if (tracebackMatrix[[i, j]] === 1) {
            align1.unshift(q);
            align2.unshift(d);
            i--; j--;
        } else if (tracebackMatrix[[i, j]] === 3) {
            align2.unshift(d);
            align1.unshift('-');
            j--;
        }
    }
    return [align2, maxScore, covered];
}

function fastaAlgorithm({
    database, query, k = 4, n = 3, bandWidth = 5, matchScore = 3, mismatchPenalty = 3, 
    gapPenalty = 1, matchProcessing = null, dpProcessing = null, 
    dbTokenDepthRoc = null, queryTokenDepthRoc = null
}) {
    const matches = ktupleMatching(query, database, k, matchProcessing);
    const topDiagonal = findTopDiagonal(matches, query, database, k);
    if (topDiagonal === null) return [null, -Infinity];

    const [bestAlignment, bestScore, covered] = smithWatermanDiagonal({
        query, database, matchScore, mismatchPenalty, gapPenalty, 
        processing: dpProcessing, diag: topDiagonal, bandWidth, 
        dbTokenDepthRoc, queryTokenDepthRoc
    });

    if (!bestAlignment) return [null, -Infinity];
    return [bestAlignment, bestScore];
}

function fuzzyFind(query, database) {
    const linewiseTokenizedDatabase = linewiseTokenize(database);
    const linewiseTokenizedQuery = linewiseTokenize(query);

    let [result, score] = fastaAlgorithm({
        database: linewiseTokenizedDatabase,
        query: linewiseTokenizedQuery,
        k: 1,
        n: 1,
        bandWidth: 10,
        matchScore: 3,
        mismatchPenalty: 3,
        gapPenalty: 0.1,
        matchProcessing: (x) => x.trim().replace(/ /g, ''),
        dpProcessing: (x) => wordwiseTokenize(x.trim().replace(/ /g, ''))
    });

    const wordwiseTokenizedDatabase = result ? wordwiseTokenize(result.join('')) : wordwiseTokenize(database);
    const wordwiseTokenizedQuery = wordwiseTokenize(query);

    [result, score] = fastaAlgorithm({
        database: wordwiseTokenizedDatabase,
        query: wordwiseTokenizedQuery,
        k: Math.max(1, Math.min(3, wordwiseTokenizedQuery.length - 1, wordwiseTokenizedDatabase.length - 1)),
        n: 1,
        bandWidth: 18,
        matchScore: 5,
        mismatchPenalty: 3,
        gapPenalty: 0.3,
        matchProcessing: (x) => x.trim(),
        dpProcessing: (x) => x.trim()
    });

    if (!result) return [null, -Infinity];
    return [result.join(''), score];
}

function getDiffs(a, b) {
    const diffs = Array.from(Diff.diffLines(a, b));
    let diffsGroups = [];
    let currentGroup = [];
    diffs.forEach(diff => {
        if (!diff.added && !diff.removed) {
            if (currentGroup.length > 0) {
                diffsGroups.push(currentGroup);
                currentGroup = [];
            }
        } else {
            currentGroup.push(diff);
        }
    });
    if (currentGroup.length > 0) {
        diffsGroups.push(currentGroup);
    }

    return diffsGroups.map(group => {
        const find = group.filter(line => line.removed).map(line => line.value).join('\n');
        const replace = group.filter(line => line.added).map(line => line.value).join('\n');
        return [find, replace];
    });
}

function parseDiff(diffString) {
    const lines = diffString.split('\n');
    let groups = [];
    let currentGroup = [];
    let inEdit = false;

    lines.forEach(line => {
        if (line.startsWith('+') || line.startsWith('-')) {
            if (!inEdit && currentGroup.length > 0) {
                groups.push(currentGroup);
                currentGroup = [];
            }
            currentGroup.push(line);
            inEdit = true;
        } else {
            if (inEdit && currentGroup.length > 0) {
                groups.push(currentGroup);
                currentGroup = [];
            }
            currentGroup.push(line);
            inEdit = false;
        }
    });
    if (currentGroup.length > 0) {
        groups.push(currentGroup);
    }

    return groups.map(group => {
        if (!(group[0].startsWith('+') || group[0].startsWith('-'))) {
            return group.join('\n');
        } else {
            let deleteLines = '';
            let insertLines = '';
            group.forEach(line => {
                if (line.startsWith('-')) deleteLines += line.slice(1) + '\n';
                if (line.startsWith('+')) insertLines += line.slice(1) + '\n';
            });
            return [deleteLines, insertLines];
        }
    });
}

function skwonk(database, diff) {
    const hunks = diff.split(/@@.*?@@/);
    let i = 0;

    for (let hunk of hunks) {
        const original = hunk.split('\n').filter(line => !line.startsWith('+')).join('\n');
        const [zone, ] = fuzzyFind(original, database.slice(i));
        i = database.indexOf(zone) + zone.length;
        const originalZone = zone;

        if (!zone) return database;

        const diffGroups = parseDiff(hunk);
        diffGroups.forEach((group, idx) => {
            if (!Array.isArray(group)) return;
            let [find, replace] = group;

            if (find === '') {
                const priorContext = fuzzyFind(diffGroups[idx - 1], zone)[0] || null;
                const furtherContext = fuzzyFind(diffGroups[idx + 1], zone)[0] || null;
                let insertionIndex = priorContext ? zone.indexOf(priorContext) + priorContext.length : null;
                if (insertionIndex === null) {
                    insertionIndex = furtherContext ? zone.indexOf(furtherContext) : 0;
                }
                zone = zone.slice(0, insertionIndex) + '\n' + replace + zone.slice(insertionIndex);
            } else {
                const [fuzz, ] = fuzzyFind(find, zone);
                if (fuzz !== null) {
                    zone = zone.replace(fuzz, replace);
                }
            }
        });

        database = database.replace(originalZone, zone);
    }
    return database;
}

function findMostSimilarFileName(files, query) {
    return files.reduce((bestMatch, file) => {
        const score = fastaAlgorithm({database: file, query, k: 6})[1];
        return score > bestMatch[1] ? [file, score] : bestMatch;
    }, [null, -Infinity])[0];
}

function computeNestedLevels(codeLines, indentIsRelevant = true) {
    let nestedLevels = [];
    let braceLevel = 0, parenLevel = 0, indentLevel = 0;
    let currentLevel = 0;
    let prevIndentSpaces = 0;

    codeLines.forEach(line => {
        if (line.trim() === '') {
            nestedLevels.push(currentLevel);
            return;
        }

        const strippedLine = line.trimEnd();
        braceLevel += (strippedLine.match(/{/g) || []).length;
        braceLevel -= (strippedLine.match(/}/g) || []).length;
        parenLevel += (strippedLine.match(/\(/g) || []).length;
        parenLevel -= (strippedLine.match(/\)/g) || []).length;

        if (indentIsRelevant) {
            const leadingSpaces = line.length - line.trimStart().length;
            indentLevel += Math.sign(leadingSpaces - prevIndentSpaces);
            prevIndentSpaces = leadingSpaces;
        }

        if (braceLevel < 0) braceLevel = 0;
        if (parenLevel < 0) parenLevel = 0;

        currentLevel = braceLevel + parenLevel + indentLevel;
        nestedLevels.push(currentLevel);
    });

    return nestedLevels;
}

module.exports = {
    arrFromSepString,
    arrFromNumberedList,
    extractMarkdownBlocks,
    markdownToDict,
    wordwiseTokenize,
    linewiseTokenize,
    editDistance,
    score,
    weightedScore,
    fuzzyFind,
    skwonk,
    findMostSimilarFileName,
    computeNestedLevels
};
