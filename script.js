const N = 100;
const NOISE_VAR = 0.05;
// Globales Chart-Register
const chartRefs = {};

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

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 })); // Linear output
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    return model;
}

async function trainAndPredict(model, trainData, testData, epochs) {
    const { xs: xsTrain, ys: ysTrain } = prepareTensors(trainData);
    const { xs: xsTest, ys: ysTest } = prepareTensors(testData);

    await model.fit(xsTrain, ysTrain, {
        epochs: epochs,
        batchSize: 32,
        verbose: 0
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
    const data = generateData();

    // Clean model
    const modelClean = createModel();
    const resultClean = await trainAndPredict(modelClean, data.trainClean, data.testClean, 100);

    // Best fit (rauschig, moderate Ep.)
    const modelBestFit = createModel();
    const resultBest = await trainAndPredict(modelBestFit, data.trainNoisy, data.testNoisy, 100);

    // Overfit (rauschig, zu viele Ep.)
    const modelOverfit = createModel();
    const resultOverfit = await trainAndPredict(modelOverfit, data.trainNoisy, data.testNoisy, 500);

    // Daten und Ergebnisse rendern
    renderAll({
        data,
        resultClean,
        resultBest,
        resultOverfit
    });
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

// Visualisierung aller Schritte
function renderAll({ data, resultClean, resultBest, resultOverfit }) {
    // R1 - Datensätze
    renderScatterChart("chartRawClean", [
        {
            label: 'Train Clean',
            data: data.trainClean,
            backgroundColor: 'blue'
        },
        {
            label: 'Test Clean',
            data: data.testClean,
            backgroundColor: 'lightblue'
        }
    ], 'R1: Ohne Rauschen');

    renderScatterChart("chartRawNoisy", [
        {
            label: 'Train Noisy',
            data: data.trainNoisy,
            backgroundColor: 'red'
        },
        {
            label: 'Test Noisy',
            data: data.testNoisy,
            backgroundColor: 'orange'
        }
    ], 'R1: Mit Rauschen');

    // R2 - Clean Model Predictions
    renderScatterChart("chartCleanTrain", [
        { label: 'Train Clean', data: data.trainClean, backgroundColor: 'blue' },
        { label: 'Prediction', data: resultClean.predictions, borderColor: 'black', type: 'line', fill: false }
    ], `R2 Train | Loss: ${resultClean.trainLoss.toFixed(4)}`);

    renderScatterChart("chartCleanTest", [
        { label: 'Test Clean', data: data.testClean, backgroundColor: 'lightblue' },
        { label: 'Prediction', data: resultClean.predictions, borderColor: 'black', type: 'line', fill: false }
    ], `R2 Test | Loss: ${resultClean.testLoss.toFixed(4)}`);

    // R3 - Best-Fit Model
    renderScatterChart("chartBestTrain", [
        { label: 'Train Noisy', data: data.trainNoisy, backgroundColor: 'red' },
        { label: 'Prediction', data: resultBest.predictions, borderColor: 'green', type: 'line', fill: false }
    ], `R3 Train | Loss: ${resultBest.trainLoss.toFixed(4)}`);

    renderScatterChart("chartBestTest", [
        { label: 'Test Noisy', data: data.testNoisy, backgroundColor: 'orange' },
        { label: 'Prediction', data: resultBest.predictions, borderColor: 'green', type: 'line', fill: false }
    ], `R3 Test | Loss: ${resultBest.testLoss.toFixed(4)}`);

    // R4 - Overfit Model
    renderScatterChart("chartOverfitTrain", [
        { label: 'Train Noisy', data: data.trainNoisy, backgroundColor: 'red' },
        { label: 'Prediction', data: resultOverfit.predictions, borderColor: 'purple', type: 'line', fill: false }
    ], `R4 Train | Loss: ${resultOverfit.trainLoss.toFixed(4)}`);

    renderScatterChart("chartOverfitTest", [
        { label: 'Test Noisy', data: data.testNoisy, backgroundColor: 'orange' },
        { label: 'Prediction', data: resultOverfit.predictions, borderColor: 'purple', type: 'line', fill: false }
    ], `R4 Test | Loss: ${resultOverfit.testLoss.toFixed(4)}`);
}

document.getElementById("generateBtn").addEventListener("click", () => {
    tf.disposeVariables();
    run();
});

run(); // Initialer Lauf
