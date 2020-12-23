<template>
  <v-app light>
    <v-app-bar app color="primary" dark>
      <div class="d-flex align-center">
        <v-img
          alt="Vuetify Logo"
          class="shrink mr-2"
          contain
          :src="require('./assets/hawk-5.svg')"
          transition="scale-transition"
          width="40"
        />

        <h3 >
          Birds of Prey Identifier
        </h3>
      </div>

      <v-spacer></v-spacer>

      <v-btn
        href="https://github.com/NomanTrips/bird-classifier"
        target="_blank"
        text
      >
        <span class="mr-2">Source code</span>
        <v-icon>mdi-open-in-new</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <MainPage />
    </v-main>
  </v-app>
</template>

<script>
import MainPage from "./components/MainPage.vue";

export default {
  components: {
    MainPage,
  },
  data() {
    return {};
  },
  methods: {},
};
</script>
