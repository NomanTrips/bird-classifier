<template>
  <v-container>
    <v-row class="text-center">
      <v-col cols="12">
        <v-img
          :src="require('../assets/hawk-profile-pic.png')"
          class="my-3"
          contain
          height="200"
        />
      </v-col>

      <v-col class="mb-5" cols="12">
        <v-file-input
          label="Upload bird of prey photo"
          filled
          prepend-icon="mdi-camera"
          v-on:change="onFileInputChange"
          v-model="file"
          outlined
        ></v-file-input>
        <v-btn
          color="primary"
          v-bind:disabled="classifyBtnEnabled"
          v-bind:loading="isLoading"
          v-on:click="submit"
        >
          Classify
        </v-btn>
      </v-col>
    </v-row>
    <v-card v-show="displayResults" class="mx-auto" max-width="90%">
      <v-list-item three-line>
        <v-list-item-content>
          <div class="overline mb-4">CLASSIFICATION RESULTS</div>
          <v-list-item-title class="headline mb-1">
            Red-Tailed Hawk
          </v-list-item-title>
          <v-list-item-subtitle>Buteo jamaicensis</v-list-item-subtitle>
        </v-list-item-content>

        <v-list-item-avatar size="100">
          <img src="../assets/red-tail-avatar.png" alt="Red-Tailed Hawk" />
        </v-list-item-avatar>
      </v-list-item>

      <v-divider class="mx-4"></v-divider>
      <v-card-text class="text--primary">
        <div
          id="app"
          class="d-flex align-center"
          style="width: 90%; height: 80%; margin: auto"
        >
          <canvas id="planet-chart"></canvas>
        </div>
        <div></div>

        <div>
          {{predictedClass}}
        </div>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script>
import Chart from "chart.js";
import planetChartData from "../chart-data.js";
import axios from 'axios'

export default {
  name: "app",
  data() {
    return {
      classifyBtnEnabled: true,
      isLoading: false,
      planetChartData: planetChartData,
      displayResults: false,
      file: null,
      predictedClass: '',
    };
  },
  methods: {
    onFileInputChange: function () {
      //this.file = File[];
      this.classifyBtnEnabled = false;
    },
    processImage: function () {
      this.isLoading = true;
      setTimeout(this.mockAsync, 3000);
    },
    mockAsync: function () {
      this.isLoading = false;
      this.displayResults = true;
    },
    submit () {
      //var reader = new FileReader();
      //reader.readAsText(this.file);
      //reader.onload = () => {
      //  this.data = reader.result;
      //}
      axios.post('http://127.0.0.1:5000/predict', {
        file : this.file,
      })
      .then((response) => {
        this.predictedClass = response.data.class_name
      })
    },
    createChart(chartId, chartData) {
      const ctx = document.getElementById(chartId);
      // eslint-disable-next-line no-unused-vars
      const myChart = new Chart(ctx, {
        type: chartData.type,
        data: chartData.data,
        options: chartData.options,
      });
    },
  },
  mounted() {
    this.createChart("planet-chart", this.planetChartData);
  },
};
</script>
