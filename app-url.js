async function loadJSON(path) {
  const response = await fetch(path);
  return await response.json();
}

// 単純なテキストをベクトル化（Bag of Words風の手法）
function tokenizeMessages(messages, vocab) {
  return messages.map(msg => {
    const tokens = Array(vocab.length).fill(0);
    msg.toLowerCase().split(/[\s!?。、「」]/).forEach(word => {
      const index = vocab.indexOf(word);
      if (index >= 0) tokens[index] = 1;
    });
    return tokens;
  });
}

// 語彙を構築
function buildVocabulary(messages) {
  const vocabSet = new Set();
  messages.forEach(msg => {
    msg.toLowerCase().split(/[\s!?。、「」]/).forEach(word => {
      if (word) vocabSet.add(word);
    });
  });
  return Array.from(vocabSet);
}

async function main() {
  const trainingData = await loadJSON('./json/training_data.json');
  const actualData = await loadJSON('./json/actual_data.json');

  const trainingMessages = trainingData.map(d => d.message);
  const trainingLabels = trainingData.map(d => parseFloat(d.waitTime) || 0);

  const actualMessages = actualData.map(d => d.message);

  const vocab = buildVocabulary(trainingMessages);
  const xs = tf.tensor2d(tokenizeMessages(trainingMessages, vocab));
  const ys = tf.tensor2d(trainingLabels, [trainingLabels.length, 1]);

  // モデルの定義
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [vocab.length] }));
  model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

  // モデルの学習
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 4,
    shuffle: true
  });

  // 実データで予測
  const actualInputs = tf.tensor2d(tokenizeMessages(actualMessages, vocab));
  const predictions = model.predict(actualInputs);
  const predictedValues = await predictions.array();

  // Chart.js で可視化
  const ctx = document.getElementById('waitChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: actualMessages,
      datasets: [{
        label: '予測待ち時間（分）',
        data: predictedValues.map(p => p[0]),
        backgroundColor: 'rgba(75, 192, 192, 0.6)'
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: '待ち時間（分）' }
        },
        x: {
          title: { display: true, text: 'メッセージ' }
        }
      }
    }
  });
}

main();
