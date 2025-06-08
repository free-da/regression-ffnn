const N = 100;
const NOISE_VAR = 0.05;
// Globales Chart-Register
const chartRefs = {};
let currentData = null;
let modelClean = null;
let modelNoisy = null;
let modelOverfit = null;
let resultClean = null;
let resultNoisy = null;
let resultOverfit = null;

// Ziel-Funktion (Ground Truth)
function targetFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// Gaussian Noise Generator
function gaussianNoise(mean = 0, variance = NOISE_VAR) {
    const u1 = Math.random();
    const u2 = Math.random();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + Math.sqrt(variance) * randStdNormal;
}

// Generiere Datensatz
function generateData() {
    const data = Array.from({ length: N }, () => {
        const x = Math.random() * 4 - 2;
        const yTrue = targetFunction(x);
        const yNoisy = yTrue + gaussianNoise();
        return { x, yTrue, yNoisy };
    });

    const shuffled = data.sort(() => Math.random() - 0.5);
    const train = shuffled.slice(0, N / 2);
    const test = shuffled.slice(N / 2);

    return {
        trainClean: train.map(d => ({ x: d.x, y: d.yTrue })),
        testClean: test.map(d => ({ x: d.x, y: d.yTrue })),
        trainNoisy: train.map(d => ({ x: d.x, y: d.yNoisy })),
        testNoisy: test.map(d => ({ x: d.x, y: d.yNoisy }))
    };
}

function prepareTensors(data) {
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));
    return { xs, ys };
}

function createModel(numberOfUnits) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: numberOfUnits, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: numberOfUnits, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 })); // Linear output
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    return model;
}

async function trainAndPredict(model, trainData, testData, epochs, label) {
    const { xs: xsTrain, ys: ysTrain } = prepareTensors(trainData);
    const { xs: xsTest, ys: ysTest } = prepareTensors(testData);

    await model.fit(xsTrain, ysTrain, {
        epochs: epochs,
        batchSize: 32,
        verbose: 0,
        callbacks: tfvis.show.fitCallbacks(
            { name: label },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    const trainLoss = model.evaluate(xsTrain, ysTrain).dataSync()[0];
    const testLoss = model.evaluate(xsTest, ysTest).dataSync()[0];

    const xPred = tf.linspace(-2, 2, 200).reshape([200, 1]);
    const yPred = model.predict(xPred);

    const xPredData = await xPred.array();
    const yPredData = await yPred.array();

    tf.dispose([xsTrain, ysTrain, xsTest, ysTest, xPred, yPred]);

    return {
        trainLoss,
        testLoss,
        predictions: xPredData.map((x, i) => ({ x: x[0], y: yPredData[i][0] }))
    };
}

// Hauptfunktion
async function run() {
    currentData = generateData()
    renderDataPreviewAndScatterChart(currentData,'dataPreview');
    // Clean model
    const resultClean = await createAndTrainModelClean();
    renderModelClean(resultClean);
    // Best fit (rauschig, moderate Ep.)
    const resultNoisy = await createAndTrainModelNoisy()
    renderModelNoisy(resultNoisy);
    // Overfit (rauschig, zu viele Ep.)
    const resultOverfit = await createAndTrainModelOverfit()
    renderModelOverfit(resultOverfit);
}

// Chart-Rendering (Chart.js)
function renderScatterChart(canvasId, datasets, title) {
    if (chartRefs[canvasId]) {
        chartRefs[canvasId].destroy();
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    chartRefs[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: title } },
            scales: {
                x: { title: { display: true, text: 'x' } },
                y: { title: { display: true, text: 'y' } }
            }
        }
    });
}

function renderDataPreviewAndScatterChart(data, elementId) {
    const previewElement = document.getElementById(elementId);
    if (!previewElement) return;
    const combined = [...data.trainNoisy, ...data.testNoisy];
    const preview = combined.slice(0, 10).map((d, i) => `${i + 1}. x: ${d.x.toFixed(3)}, y: ${d.y.toFixed(3)}`).join("\n");
    previewElement.textContent = '';
    previewElement.textContent = preview;
    renderScatterChart("chartRawClean", [
        {
            label: 'Train Clean',
            data: currentData.trainClean,
            backgroundColor: 'blue'
        },
        {
            label: 'Test Clean',
            data: currentData.testClean,
            backgroundColor: 'lightblue'
        }
    ], 'R1: Ohne Rauschen');

    renderScatterChart("chartRawNoisy", [
        {
            label: 'Train Noisy',
            data: currentData.trainNoisy,
            backgroundColor: 'red'
        },
        {
            label: 'Test Noisy',
            data: currentData.testNoisy,
            backgroundColor: 'orange'
        }
    ], 'R1: Mit Rauschen');
}

document.getElementById("generateBtn").addEventListener("click", () => {
    tf.disposeVariables();
    const timestamp = new Date().toLocaleString();
    document.getElementById('dataPreview').textContent = "Neu erzeugt: " + timestamp;
    currentData = generateData();
    renderDataPreviewAndScatterChart(currentData,'dataPreview');
});

document.getElementById("downloadDataset").addEventListener("click", () => {
    if (!currentData) {
        alert("Bitte zuerst einen Datensatz generieren.");
        return;
    }
    const blob = new Blob([JSON.stringify(currentData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.download = `dataset-${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);
});
function uploadDataset(callback) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json";
    input.onchange = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = () => {
            const json = JSON.parse(reader.result);
            callback(json);
        };
        reader.readAsText(file);
    };
    input.click();
}

document.getElementById("uploadDataset").addEventListener("click", () => {
    uploadDataset((json) => {
        currentData = json;
        renderDataPreviewAndScatterChart(currentData, 'dataPreview');
        console.log("Datensatz erfolgreich geladen:", currentData);
    });
});

function renderModelClean(resultClean) {
    renderScatterChart("chartCleanTrain", [
        {label: 'Train Clean', data: currentData.trainClean, backgroundColor: 'blue'},
        {label: 'Prediction', data: resultClean.predictions, borderColor: 'black', type: 'line', fill: false}
    ], `R2 Train | Loss: ${resultClean.trainLoss.toFixed(4)}`);
    renderScatterChart("chartCleanTest", [
        {label: 'Test Noisy', data: currentData.testNoisy, backgroundColor: 'orange'},
        {label: 'Prediction', data: resultClean.predictions, borderColor: 'green', type: 'line', fill: false}
    ], `R3 Test | Loss: ${resultClean.testLoss.toFixed(4)}`);
}

async function createAndTrainModelClean() {
    const units = parseInt(document.getElementById("unitsInputClean").value);
    const epochs = parseInt(document.getElementById("epochsInputClean").value);
    tfvis.visor().open();
    modelClean = createModel(units);
    resultClean = await trainAndPredict(modelClean, currentData.trainClean, currentData.testClean, epochs, "Unverrauscht");
    return resultClean;
}

function renderModelNoisy(resultNoisy) {
    renderScatterChart("chartBestTrain", [
        {label: 'Train Noisy', data: currentData.trainNoisy, backgroundColor: 'red'},
        {label: 'Prediction', data: resultNoisy.predictions, borderColor: 'green', type: 'line', fill: false}
    ], `R3 Train | Loss: ${resultNoisy.trainLoss.toFixed(4)}`);

    renderScatterChart("chartBestTest", [
        {label: 'Test Noisy', data: currentData.testNoisy, backgroundColor: 'orange'},
        {label: 'Prediction', data: resultNoisy.predictions, borderColor: 'green', type: 'line', fill: false}
    ], `R3 Test | Loss: ${resultNoisy.testLoss.toFixed(4)}`);
}

async function createAndTrainModelNoisy() {
    const units = parseInt(document.getElementById("unitsInputNoisy").value);
    const epochs = parseInt(document.getElementById("epochsInputNoisy").value);
    tfvis.visor().open();
    modelNoisy = createModel(units);
    resultNoisy = await trainAndPredict(modelNoisy, currentData.trainNoisy, currentData.testNoisy, epochs, "Mit Rauschen");
    return resultNoisy;
}

function renderModelOverfit(resultOverfit) {
    renderScatterChart("chartOverfitTrain", [
        {label: 'Train Noisy', data: currentData.trainNoisy, backgroundColor: 'red'},
        {label: 'Prediction', data: resultOverfit.predictions, borderColor: 'purple', type: 'line', fill: false}
    ], `R4 Train | Loss: ${resultOverfit.trainLoss.toFixed(4)}`);

    renderScatterChart("chartOverfitTest", [
        {label: 'Test Noisy', data: currentData.testNoisy, backgroundColor: 'orange'},
        {label: 'Prediction', data: resultOverfit.predictions, borderColor: 'purple', type: 'line', fill: false}
    ], `R4 Test | Loss: ${resultOverfit.testLoss.toFixed(4)}`);
}

async function createAndTrainModelOverfit() {
    const units = parseInt(document.getElementById("unitsInputOverfit").value);
    const epochs = parseInt(document.getElementById("epochsInputOverfit").value);
    tfvis.visor().open();
    modelOverfit = createModel(units);
    resultOverfit = await trainAndPredict(modelOverfit, currentData.trainNoisy, currentData.testNoisy, epochs, "Overfit");
    return resultOverfit;
}

document.getElementById("retrainClean").addEventListener("click", async () => {
    tfvis.visor().open();
    const resultClean = await createAndTrainModelClean();
    renderModelClean(resultClean);
});
document.getElementById("retrainNoisy").addEventListener("click", async () => {
    tfvis.visor().open();
    const resultNoisy = await createAndTrainModelNoisy()
    renderModelNoisy(resultNoisy);
});
document.getElementById("retrainOverfit").addEventListener("click", async () => {
    tfvis.visor().open();
    const resultOverfit = await createAndTrainModelOverfit()
    renderModelOverfit(resultOverfit);
});

function getModelByType(type) {
    switch (type) {
        case "clean":
            return modelClean;
        case "noisy":
            return modelNoisy;
        case "overfit":
            return modelOverfit;
        default:
            throw new Error(`Unbekannter Modelltyp: ${type}`);
    }
}

async function loadSavedDataAndModels() {
    await loadDatasetFromFile();

    modelClean = await loadModelByType('clean');
    modelNoisy = await loadModelByType('noisy');
    modelOverfit = await loadModelByType('overfit');
}

function renderAllFromLoadedData() {
    renderDataPreviewAndScatterChart(currentData,'dataPreview');
    // Clean model
    renderModelClean(resultClean);
    // Best fit (rauschig, moderate Ep.)
    renderModelNoisy(resultNoisy);
    // Overfit (rauschig, zu viele Ep.)
    renderModelOverfit(resultOverfit);
}



async function loadDatasetFromFile() {
    try {
        const response = await fetch('dataset.json');
        if (!response.ok) throw new Error("Datensatz nicht gefunden");

        currentData = await response.json();

        console.log("Datensatz erfolgreich geladen.");
    } catch (err) {
        console.error("Fehler beim Laden des Datensatzes:", err);
    }
}

async function loadModelByType(type) {
    try {
        const model = await tf.loadLayersModel(`model-${type}.json`);
        console.log(`Modell '${type}' geladen.`);
        return model;
    } catch (err) {
        console.error(`Fehler beim Laden des Modells '${type}':`, err);
        return null;
    }
}

async function saveModel(type) {
    const model = getModelByType(type);
    if (!model) {
        console.error(`Kein Modell vom Typ '${type}' gefunden.`);
        return;
    }

    // Modell speichern
    await model.save(`downloads://model_${type}`);

    // Trainingsdaten holen
    const result = getResultByType(type);

    if (!result) {
        console.error(`Keine Trainingsdaten für Modell '${type}' gefunden.`);
        return;
    }

    // Exportiere relevante Trainingsdaten in eine Datei
    const trainingData = {
        trainLoss: result.trainLoss,
        testLoss: result.testLoss,
        trainPredictions: result.trainPredictions,
        testPredictions: result.testPredictions,
        xTrain: result.xTrain, // optional
        xTest: result.xTest,   // optional
        yTrain: result.yTrain,
        yTest: result.yTest,
        units: result.units,
        epochs: result.epochs
    };

    const blob = new Blob([JSON.stringify(trainingData)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_${type}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function getResultByType(type) {
    switch (type) {
        case "clean":
            return resultClean;
        case "noisy":
            return resultNoisy;
        case "overfit":
            return resultOverfit;
        default:
            console.error(`Unbekannter Modelltyp: ${type}`);
            return null;
    }
}

window.addEventListener('load', async () => {
    await loadSavedDataAndModels();
    renderAllFromLoadedData();
});

document.getElementById("generateBtn").addEventListener("click",run);
document.getElementById("downloadClean").addEventListener("click", () => saveModel("clean"));
document.getElementById("downloadNoisy").addEventListener("click", () => saveModel("noisy"));
document.getElementById("downloadOverfit").addEventListener("click", () => saveModel("overfit"));

//run();
