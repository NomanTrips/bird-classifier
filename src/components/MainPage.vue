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
          Identify
        </v-btn>
      </v-col>
    </v-row>
    <v-card v-show="displayResults" class="mx-auto" max-width="300px">
      <v-list-item three-line>
        <v-list-item-content>
          <div class="overline mb-4">IDENTIFICATION RESULTS</div>
          <v-list-item-title class="headline mb-1">
            {{ predictedClass.name }}
          </v-list-item-title>
          <v-list-item-subtitle></v-list-item-subtitle>
        </v-list-item-content>
      </v-list-item>

      <v-divider class="mx-4"></v-divider>
      <v-img
        :src="getImgUrl()"
        max-height="500"
        max-width="300"
      ></v-img>
    </v-card>
  </v-container>
</template>

<script>
import axios from "axios";
import _ from "lodash";

export default {
  name: "app",
  data() {
    return {
      classifyBtnEnabled: true,
      isLoading: false,
      displayResults: false,
      file: null,
      predictedClass: {
        id: 0,
        name: "",
        pictureUrl: "red-tailed-hawk-avatar",
        model_id: "",
      },
      data: "",
      classes: [
        {
          id: 0,
          name: "Unknown",
          pictureUrl: "unknown-avatar",
          model_id: "northern-goshawk",
        },
        {
          id: 1,
          name: "Red-Tailed Hawk",
          pictureUrl: "red-tailed-hawk-avatar",
          model_id: "red-tailed-hawk",
        },
        {
          id: 2,
          name: "Vulture",
          pictureUrl: "vulture-avatar",
          model_id: "vulture",
        },
        {
          id: 3,
          name: "Coopers Hawk",
          pictureUrl: "coopers-hawk-avatar",
          model_id: "coopers-hawk",
        },
        {
          id: 4,
          name: "Osprey",
          pictureUrl: "osprey-avatar",
          model_id: "osprey",
        },
        {
          id: 5,
          name: "Bald Eagle",
          pictureUrl: "eagle-avatar",
          model_id: "bald-eagle",
        },
        {
          id: 6,
          name: "Barred Owl",
          pictureUrl: "barred-owl-avatar",
          model_id: "barred-owl",
        },
        {
          id: 7,
          name: "Crow",
          pictureUrl: "crow-avatar",
          model_id: "crow",
        },
        {
          id: 8,
          name: "Peregrine Falcon",
          pictureUrl: "falcon-avatar",
          model_id: "peregrine-falcon",
        },
        {
          id: 9,
          name: "Northern Goshawk",
          pictureUrl: "goshawk-avatar",
          model_id: "northern-goshawk",
        },
        {
          id: 10,
          name: "American Kestrel",
          pictureUrl: "kestrel-avatar",
          model_id: "american-kestrel",
        },
        {
          id: 11,
          name: "Great horned owl",
          pictureUrl: "great-horned-owl-avatar",
          model_id: "great-horned-owl",
        },
      ],
    };
  },
  methods: {
    getImgUrl: function () {
      var pic = this.predictedClass.pictureUrl;
      //return require('../assets/avatars/'+pic);
      var images = require.context("../assets/avatars", false, /\.png$/);
      return images("./" + pic + ".png");
    },
    onFileInputChange: function () {
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
    submit() {
      let formData = new FormData();
      formData.append("file", this.file);
      axios
        .post("http://52.37.86.143:5000/predict", formData, {
        //  .post("http://127.0.0.1:5000/predict", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((response) => {
          //this.predictedClass = response.data.class_name;
          this.predictedClass = _.find(this.classes, function (o) {
            return o.model_id == response.data.class;
          });
          this.displayResults = true;
        })
        .catch(function () {
          console.log("FAILURE!!");
        });
    },
  },
  mounted() {
  },
};
</script>
