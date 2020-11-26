export const planetChartData = {
    type: 'bar',
    data: {
      labels: ['Red-Tailed Hawk', 'Bald Eagle', 'Osprey', 'Vulture', 'Crow', 'Barred Owl', 'Falcon', 'Coopers Hawk'],
      datasets: [
        { // another line graph
          label: 'Class energy',
          data: [4.8, 12.1, 12.7, 6.7, 139.8, 116.4, 50.7, 49.2],
          backgroundColor: [
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
            'rgba(255, 203, 5, 1)', // Orange
          ],
          borderColor: [
            '#FF4500',
            '#FF4500',
            '#FF4500',
            '#FF4500',
            '#FF4500',
            '#FF4500',
            '#FF4500',
          ],
          borderWidth: 3
        }
      ]
    },
    options: {
      responsive: true,
      lineTension: 1,
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: true,
            padding: 25,
          }
        }]
      }
    }
  }
  
  export default planetChartData;