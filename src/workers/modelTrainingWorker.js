import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = null;
let _model = null;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

// Normalize continuous values (price, age) to 0-1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 -> 0.56
const normalize = (value, min, max) => {
    if (max === min) return 0.5;
    return (value - min) / (max - min);
}

function makeContext(catalog, users){
    const ages = users.map(u => u.age)
    const prices = catalog.map(c => c.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(catalog.map(p => p.color))]
    const categories = [...new Set(catalog.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))
    
    // Computar a média de idade dos compradores por produto 
    // (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}
    
    users.forEach(user => {
        user.purchases?.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        catalog.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge
            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        catalog,
        users,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        midAge,
        productAvgAgeNorm,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimensions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) => {
    if (index === undefined) return tf.zeros([length]);
    return tf.oneHot(index, length).cast('float32').mul(weight);
}

function encodeProduct(product, context){
    return tf.tidy(() => {
        // normalizando dados para ficar de 0 a 1 e
        // aplicar o peso na recomendação
        const price = tf.tensor1d([
            normalize(
                product.price,
                context.minPrice,
                context.maxPrice
            ) * WEIGHTS.price
        ])

        const age = tf.tensor1d([
            (
                context.productAvgAgeNorm[product.name] ?? 0.5
            ) * WEIGHTS.age
        ])

        const category = oneHotWeighted(
            context.categoriesIndex[product.category],
            context.numCategories,
            WEIGHTS.category
        )

        const color = oneHotWeighted(
            context.colorsIndex[product.color],
            context.numColors,
            WEIGHTS.color
        )

        return tf.concat(
            [price, age, category, color]
        )
    })
}

function encodeUser(user, context) {
    return tf.tidy(() => {
        const purchases = user.purchases ?? [];

        if (purchases.length) {
            return tf.stack(
                purchases.map(
                    product => encodeProduct(product, context)
                )
            ).mean(0).reshape([
                1,
                context.dimensions
            ])
        }

        const price = tf.tensor1d([0]);
        const age = tf.tensor1d([
            normalize(
                user.age ?? context.midAge,
                context.minAge,
                context.maxAge
            ) * WEIGHTS.age
        ]);
        const categories = tf.zeros([context.numCategories]);
        const colors = tf.zeros([context.numColors]);

        return tf.concat([price, age, categories, colors]).reshape([
            1,
            context.dimensions
        ]);
    })
}

function tensorToArray(tensor) {
    const values = Array.from(tensor.dataSync());
    tensor.dispose();
    return values;
}

function createTrainingData(context){
    const inputs = []
    const labels = []

    context.users
        .filter(user => user.purchases?.length)
        .forEach(user => {
            const userVector = tensorToArray(encodeUser(user, context));

            context.catalog.forEach(product => {
                const productVector = tensorToArray(encodeProduct(product, context));
                const label = Number(
                    user.purchases.some(purchase => purchase.name === product.name)
                );

                // Combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })

    return {
        xs: tf.tensor2d(inputs, [inputs.length, context.dimensions * 2]),
        ys: tf.tensor2d(labels, [ labels.length, 1 ]),
        // tamanho = userVector + productVector
        inputDimension: context.dimensions * 2
    }
}

async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimension],
            units: 128,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída 
    // - 1 neurônio porque vamos retornar apenas uma pontuação de
    // recomendação 
    // - activation: 'sigmoid' comprime o resultado para o
    // intervalo 0-1 
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação
    // fraca
    model.add(
        tf.layers.dense({units: 1, activation: 'sigmoid'})
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const progress = Math.min(99, 50 + Math.round(((epoch + 1) / 100) * 49));
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch + 1,
                    loss: logs.loss,
                    accuracy: logs.accuracy ?? logs.acc ?? 0
                })
                postMessage({
                    type: workerEvents.progressUpdate,
                    progress: { progress }
                })
            }
        }
    })

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 10 } });
    const catalogUrl = new URL('../../data/products.json', import.meta.url);
    const catalog = await (await fetch(catalogUrl)).json()

    const context = makeContext(catalog, users)
    context.productVectors = catalog.map(product => {
        const vector = encodeProduct(product, context);
        return {
            name: product.name,
            meta: {...product},
            vector: tensorToArray(vector)
        }
    })

    _globalCtx = context
    postMessage({
        type: workerEvents.tfVisData,
        data: {
            weights: WEIGHTS,
            catalog,
            users
        }
    });
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const trainData = createTrainingData(context)
    _model?.dispose();
    _model = await configureNeuralNetAndTrain(trainData)
    trainData.xs.dispose();
    trainData.ys.dispose();

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });

}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)

    if (!_model || !ctx) {
        postMessage({
            type: workerEvents.recommend,
            user,
            recommendations: []
        });
        return;
    }

    const userVector = tensorToArray(encodeUser(user, ctx));
    const purchasedNames = new Set((user.purchases ?? []).map(product => product.name));
    const recommendations = ctx.catalog
        .filter(product => !purchasedNames.has(product.name))
        .map(product => {
            const score = tf.tidy(() => {
                const productVector = encodeProduct(product, ctx);
                const input = tf.tensor2d(
                    [[...userVector, ...Array.from(productVector.dataSync())]],
                    [1, ctx.dimensions * 2]
                );
                return _model.predict(input).dataSync()[0];
            });

            return {
                ...product,
                score
            };
        })
        .sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
