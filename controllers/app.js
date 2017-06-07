angular.module("MyBlog", [], function($interpolateProvider) {
  $interpolateProvider.startSymbol('[[');
  $interpolateProvider.endSymbol(']]');
})
  .run(function () {
    console.log("App ready!")
  })
